# viewer.py
#
# Dear PyGui viewer for GS-IR relighting with PBR shading.
# - Loads a GSIR Gaussian checkpoint
# - Loads an HDRI (latlong) and builds a cubemap light
# - Renders a single selected camera view with PBR shading
# - Displays the image in a DearPyGui window with simple controls
#
# Requirements (available in your GS-IR repo):
#   arguments.py, gaussian_renderer.py, pbr.py, scene.py, utils.*
# Plus: dearpygui, nvdiffrast.torch, torchvision, torch, numpy, opencv
#
# Example:
#   python viewer.py \
#     -m output/garden-linear/ \
#     -s dataset/nerf_data/nerf_real_360/garden/ \
#     --checkpoint output/garden-linear/chkpnt35000.pth \
#     --hdri assets/hdri/studio_small_09_2k.hdr \
#     --width 1280 --height 720 --tone --gamma

import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import dearpygui.dearpygui as dpg
import nvdiffrast.torch as dr

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import viridis_cmap

# -----------------------------
# Utilities
# -----------------------------

def read_hdr(path: str) -> np.ndarray:
    """Read a latlong HDRI into float32 RGB numpy array."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HDRI not found: {path}")
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"Failed to decode HDRI: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    else:  # s == 5
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map: torch.Tensor, res_hw: List[int]) -> torch.Tensor:
    """Convert latlong environment to a cubemap (6, H, W, C)."""
    H, W = res_hw
    C = latlong_map.shape[-1]
    cubemap = torch.zeros(6, H, W, C, dtype=torch.float32, device=latlong_map.device)
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=latlong_map.device),
            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=latlong_map.device),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)
        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)
        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode="linear")[0]
    return cubemap


def tensor_to_dpg_rgba(img: torch.Tensor) -> np.ndarray:
    """Convert [H,W,3] float tensor in [0,1] to flattened RGBA float array for DearPyGui."""
    img = img.clamp(0.0, 1.0)
    H, W, _ = img.shape
    alpha = torch.ones(H, W, 1, device=img.device, dtype=img.dtype)
    rgba = torch.cat([img, alpha], dim=-1).contiguous()  # [H,W,4]
    return rgba.detach().cpu().numpy().astype(np.float32).ravel()


# -----------------------------
# Viewer App
# -----------------------------

class RelightViewer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GS-IR scene & model
        self.gaussians = GaussianModel(args.sh_degree)
        self.scene = Scene(args, self.gaussians, shuffle=False)

        # Load checkpoint
        self._load_checkpoint(args.checkpoint)

        # Load HDRI -> Cubemap light
        if args.hdri is None:
            raise ValueError("--hdri is required for PBR rendering.")
        print(f"[viewer] Loading HDRI: {args.hdri}")
        hdri_np = read_hdr(args.hdri)
        self.hdri = torch.from_numpy(hdri_np).to(self.device)
        res = args.env_res
        self.light = CubemapLight(base_res=res).to(self.device)
        self.light.base.data = latlong_to_cubemap(self.hdri, [res, res])
        self.light.eval()
        self.light.build_mips()
        self.brdf_lut = get_brdf_lut().to(self.device)

        # Precompute canonical rays for view-dir reconstruction
        self.canonical_rays = self.scene.get_canonical_rays()

        # Cameras
        self.train_cams = self.scene.getTrainCameras()
        self.test_cams = self.scene.getTestCameras()
        self.split = "train" if len(self.train_cams) > 0 else "test"
        self.cam_index = 0

        # Render state
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        self.enable_tone = bool(args.tone)
        self.enable_gamma = bool(args.gamma)
        self.enable_metallic = bool(args.metallic)

        # DearPyGui state
        self.texture_id = None  # dynamic texture id
        self.tex_size = (0, 0)
        self.last_image_rgba: Optional[np.ndarray] = None

    def _load_checkpoint(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[viewer] Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        if isinstance(checkpoint, Tuple):
            model_params = checkpoint[0]
        elif isinstance(checkpoint, Dict):
            model_params = checkpoint.get("gaussians", checkpoint)
        else:
            raise TypeError("Unexpected checkpoint format.")
        self.gaussians.restore(model_params)

    # -------------------------
    # Rendering
    # -------------------------
    @torch.no_grad()
    def render_current(self) -> torch.Tensor:
        view = self.current_view()

        # GS-IR forward pass (request normals, albedo, roughness, metallic)
        self.background[...] = 0.0
        rendering_result = render(
            viewpoint_camera=view,
            pc=self.scene.gaussians,
            pipe=self.args.pipeline,
            bg_color=self.background,
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        normal_map = rendering_result["normal_map"]  # [3,H,W]
        normal_mask = rendering_result["normal_mask"]  # [1,H,W]
        albedo_map = rendering_result["albedo_map"]  # [3,H,W]
        roughness_map = rendering_result["roughness_map"]  # [1,H,W]
        metallic_map = rendering_result["metallic_map"]  # [1,H,W]

        # View directions per pixel (world space)
        H, W = view.image_height, view.image_width
        c2w = torch.inverse(view.world_view_transform.T)  # [4,4]
        view_dirs = -(
            (F.normalize(self.canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H,W,3]

        # Mask
        alpha_mask = view.gt_alpha_mask.to(self.device)

        # PBR shading
        result = pbr_shading(
            light=self.light,
            normals=normal_map.permute(1, 2, 0),  # [H,W,3]
            view_dirs=view_dirs,  # [H,W,3]
            mask=normal_mask.permute(1, 2, 0),  # [H,W,1]
            albedo=albedo_map.permute(1, 2, 0),  # [H,W,3]
            roughness=roughness_map.permute(1, 2, 0),  # [H,W,1]
            metallic=metallic_map.permute(1, 2, 0) if self.enable_metallic else None,  # [H,W,1]
            tone=self.enable_tone,
            gamma=self.enable_gamma,
            brdf_lut=self.brdf_lut,
        )
        render_rgb = result["render_rgb"].clamp(0.0, 1.0)  # [H,W,3]
        render_rgb = render_rgb.permute(2, 0, 1) * alpha_mask  # [3,H,W]
        render_rgb = render_rgb.permute(1, 2, 0).contiguous()  # [H,W,3]
        return render_rgb

    def current_view(self):
        cams = self.train_cams if self.split == "train" else self.test_cams
        self.cam_index = max(0, min(self.cam_index, len(cams) - 1))
        return cams[self.cam_index]

    # -------------------------
    # DearPyGui UI
    # -------------------------
    # (dynamic textures don't need a separate ensure step)

    def _update_image(self, img_rgb: torch.Tensor):
        H, W, _ = img_rgb.shape
        rgba = tensor_to_dpg_rgba(img_rgb)
        rgba_list = rgba.tolist()
        if self.texture_id is None or self.tex_size != (W, H):
            # (Re)create dynamic texture and retarget the image widget
            try:
                if self.texture_id is not None:
                    dpg.delete_item(self.texture_id)
            except Exception:
                pass
            with dpg.texture_registry(show=False):
                self.texture_id = dpg.add_dynamic_texture(W, H, rgba_list)
            self.tex_size = (W, H)
            # If image widget exists, retarget it; otherwise it will be created later
            if dpg.does_item_exist("render_image"):
                dpg.configure_item("render_image", texture_tag=self.texture_id)
        else:
            dpg.set_value(self.texture_id, rgba_list)
        self.last_image_rgba = np.array(rgba, dtype=np.float32)

    def run(self):
        dpg.create_context()

        # Pre-render once to bootstrap texture size and window sizing
        img = self.render_current()
        H, W, _ = img.shape
        rgba = tensor_to_dpg_rgba(img).tolist()
        self.tex_size = (W, H)

        dpg.create_viewport(title="GS-IR PBR Viewer", width=self.args.width, height=self.args.height)

        with dpg.texture_registry(show=False):
            self.texture_id = dpg.add_dynamic_texture(W, H, rgba)

        # UI callbacks
        def render_callback():
            try:
                img2 = self.render_current()
                self._update_image(img2)
            except Exception as e:
                print(f"[viewer] Render error: {e}")

        def on_split_change(sender, app_data):
            self.split = app_data
            self.cam_index = 0
            render_callback()

        def on_cam_change(sender, app_data):
            self.cam_index = int(app_data)
            render_callback()

        def on_toggle_tone(sender, app_data):
            self.enable_tone = bool(app_data)
            render_callback()

        def on_toggle_gamma(sender, app_data):
            self.enable_gamma = bool(app_data)
            render_callback()

        def on_toggle_metallic(sender, app_data):
            self.enable_metallic = bool(app_data)
            render_callback()

        with dpg.window(label="Controls", width=380, height=-1, pos=(10, 10)):
            dpg.add_text("Dataset & Camera")
            dpg.add_combo(items=["train", "test"], default_value=self.split, label="Split", callback=on_split_change)
            train_n = len(self.train_cams)
            test_n = len(self.test_cams)
            dpg.add_text(f"Cameras: train={train_n}, test={test_n}")
            dpg.add_input_int(label="Camera Index", default_value=0, min_value=0, step=1, callback=on_cam_change)

            dpg.add_separator()
            dpg.add_text("Shading")
            dpg.add_checkbox(label="ACES tone mapping", default_value=self.enable_tone, callback=on_toggle_tone)
            dpg.add_checkbox(label="Gamma correction (sRGB)", default_value=self.enable_gamma, callback=on_toggle_gamma)
            dpg.add_checkbox(label="Use metallic map", default_value=self.enable_metallic, callback=on_toggle_metallic)

            dpg.add_separator()
            dpg.add_button(label="Render", callback=render_callback)

            dpg.add_separator()
            dpg.add_text(f"Checkpoint:\n{self.args.checkpoint}")
            dpg.add_text(f"HDRI:\n{self.args.hdri}")

        with dpg.window(label="Render",  tag="Render", width=self.args.width - 410, height=self.args.height - 40, pos=(400, 10)):
            dpg.add_image(texture_tag=self.texture_id, tag="render_image")

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Render", True)
        dpg.start_dearpygui()
        dpg.destroy_context()


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="GS-IR PBR DearPyGui Viewer")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to GS-IR checkpoint to load.")
    parser.add_argument("--hdri", type=str, required=True, help="Path to latlong HDRI file (.hdr).")
    parser.add_argument("--env_res", type=int, default=256, help="Cubemap base resolution per face.")
    parser.add_argument("--width", type=int, default=1280, help="Viewer window width.")
    parser.add_argument("--height", type=int, default=720, help="Viewer window height.")
    parser.add_argument("--tone", action="store_true", help="Enable ACES filmic tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear->sRGB gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Use reconstructed metallic map.")

    args = get_combined_args(parser)

    # Keep a copy of pipeline params inside args for convenient access in the viewer
    class _Args:
        pass
    _a = _Args()
    _a.__dict__.update(vars(args))
    _a.pipeline = pipeline.extract(args)
    _a.sh_degree = args.sh_degree

    model_path = os.path.dirname(args.checkpoint)
    print("[viewer] Model path:", model_path)

    safe_state(getattr(args, "quiet", False))

    app = RelightViewer(_a)
    app.run()
