# gsir_dearpygui_viewer.py
#
# Dear PyGui viewer (no Scene) that:
# 1) Loads a GSIR Gaussian checkpoint
# 2) Builds a manual look-at camera (MiniCam schema: FoVx/FoVy/world_view/full_proj)
# 3) Renders and displays
# 4) Mouse orbit controls:
#    - LMB drag: orbit (yaw/pitch)
#    - RMB drag: pan (move target in view plane)
#    - Wheel: zoom (dolly radius)
#
# Example:
# python gsir_dearpygui_viewer.py \
#   --checkpoint output/garden-linear/chkpnt35000.pth \
#   --width 800 --height 600 \
#   --fov_deg 60 \
#   --eye 0 0 3 --center 0 0 0 --up 0 1 0

import os
import sys
import math
from argparse import ArgumentParser
from typing import Dict, Tuple, Union

import numpy as np
import torch
import dearpygui.dearpygui as dpg

# ------ your codebase imports ------
from arguments import PipelineParams
from gaussian_renderer import GaussianModel, render
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


# ----------------------------- Camera helpers -----------------------------

def normalize_t(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return v / (torch.linalg.norm(v) + eps)


def lookat_to_RT(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build COLMAP-style world->camera [R|T].
    x_cam = R x_world + T,   T = -R * C,   C = eye (camera center in world).
    """
    z = normalize_t(eye - center)              # forward (from center to eye)
    x = normalize_t(torch.cross(up, z))        # right
    y = torch.cross(z, x)                      # true up

    R = torch.stack([x, y, z], dim=0)          # rows
    T = -R @ eye

    return (
        R.detach().cpu().numpy().astype(np.float32),
        T.detach().cpu().numpy().astype(np.float32),
    )


def fovx_from_fovy(fovy_rad: float, aspect: float) -> float:
    # tan(fovx/2) = aspect * tan(fovy/2)
    return 2.0 * math.atan(aspect * math.tan(0.5 * fovy_rad))


class MiniCam:
    """
    Matches your MiniCam fields (FoVx/FoVy, znear/zfar, world_view_transform, full_proj_transform, image_*).
    """
    def __init__(
        self,
        width: int,
        height: int,
        fovy: float,
        fovx: float,
        znear: float,
        zfar: float,
        world_view_transform: torch.Tensor,
        projection_matrix: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        self.world_view_transform = world_view_transform
        self.projection_matrix = projection_matrix
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)
        ).squeeze(0)

        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3].to(device)


def build_minicam(
    width: int, height: int, fov_deg: float,
    eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor,
    device: torch.device,
    znear: float = 0.01, zfar: float = 100.0
) -> MiniCam:
    """
    Builds a MiniCam using your utils: getWorld2View2 and getProjectionMatrix.
    Mirrors your Camera layout:
      world_view_transform = getWorld2View2(...).T
      projection_matrix   = getProjectionMatrix(...).T
    """
    R_np, T_np = lookat_to_RT(eye, center, up)
    trans = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = 1.0

    fovy = math.radians(fov_deg)
    aspect = float(width) / float(height)
    fovx = fovx_from_fovy(fovy, aspect)

    w2v = torch.tensor(getWorld2View2(R_np, T_np, trans, scale), dtype=torch.float32, device=device).transpose(0, 1)
    proj = torch.tensor(getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy),
                        dtype=torch.float32, device=device).transpose(0, 1)

    return MiniCam(width, height, fovy, fovx, znear, zfar, w2v, proj, device)


# ----------------------------- Image utils -----------------------------

def tensor_to_rgba_list(img: Union[torch.Tensor, np.ndarray]) -> Tuple[int, int, list]:
    """
    Convert a [3,H,W] float tensor in [0,1] (or np array [H,W,3]) to an RGBA float list for DearPyGui.
    Returns: (width, height, flat_list_rgba)
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
    else:
        img = np.clip(img, 0.0, 1.0)

    h, w, _ = img.shape
    alpha = np.ones((h, w, 1), dtype=img.dtype)
    rgba = np.concatenate([img, alpha], axis=-1)  # [H,W,4] in [0,1]
    return w, h, rgba.reshape(-1).tolist()


def make_bg_for_renderer(H: int, W: int, bg_color_val: float, device: torch.device) -> torch.Tensor:
    # try [3], fall back to [3,H,W] in render_view
    return torch.tensor([bg_color_val, bg_color_val, bg_color_val],
                        dtype=torch.float32, device=device)


# ---------------------------- Loader & Render -----------------------------

@torch.no_grad()
def load_gaussians_from_ckpt(checkpoint_path: str, sh_degree: int, device: torch.device) -> GaussianModel:
    gaussians = GaussianModel(sh_degree)
    ckpt = torch.load(
        checkpoint_path,
        map_location=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if isinstance(ckpt, tuple):
        model_params = ckpt[0]
    elif isinstance(ckpt, dict):
        model_params = ckpt.get("gaussians", ckpt.get("state_dict", ckpt))
    else:
        raise TypeError("Unsupported checkpoint format for GSIR checkpoint.")

    gaussians.restore(model_params)
    return gaussians


@torch.no_grad()
def render_view(
    cam: MiniCam,
    gaussians: GaussianModel,
    pipeline,
    bg_color_val: float = 0.0,
) -> torch.Tensor:
    device = cam.world_view_transform.device
    H, W = cam.image_height, cam.image_width
    bg_vec = make_bg_for_renderer(H, W, bg_color_val, device)

    try:
        out: Dict[str, torch.Tensor] = render(
            viewpoint_camera=cam,
            pc=gaussians,
            pipe=pipeline,
            bg_color=bg_vec,      # [3]
            inference=True,
            derive_normal=False,
        )
    except Exception:
        bg_img = torch.full((3, H, W), fill_value=bg_color_val,
                            dtype=torch.float32, device=device)
        out: Dict[str, torch.Tensor] = render(
            viewpoint_camera=cam,
            pc=gaussians,
            pipe=pipeline,
            bg_color=bg_img,      # [3,H,W]
            inference=True,
            derive_normal=False,
        )

    return out["render"]  # [3,H,W] in [0,1]


# ----------------------------- GUI App ------------------------------

class GSIRViewerApp:
    def __init__(self, cam: MiniCam, gaussians: GaussianModel, pipeline, bg: float = 0.0):
        self.cam = cam
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.texture_id = None
        self.tex_width = 0
        self.tex_height = 0
        self.bg = bg

        # Orbit state
        self.device = self.cam.world_view_transform.device
        self.target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        # derive yaw/pitch/radius from initial eye/target
        v = (self.cam.camera_center - self.target).detach().cpu().numpy()
        self.radius = float(np.linalg.norm(v) + 1e-9)
        # yaw: angle around +Y; pitch: elevation from horizon
        self.yaw = math.atan2(v[0], v[2])                   # [-pi, pi]
        self.pitch = math.atan2(v[1], math.sqrt(v[0]**2 + v[2]**2))  # (-pi/2, pi/2)

        # interaction
        self._last_mouse_pos = (0.0, 0.0)
        self._lmb_down = False
        self._rmb_down = False

        # sensitivities
        self.rotate_sensitivity = 0.0005     # radians per pixel
        self.pan_sensitivity = 0.00015       # world units per pixel (scaled by radius)
        self.zoom_sensitivity = 0.01         # wheel units -> radius scale

    # ---------- camera rebuild ----------

    def _spherical_to_eye(self) -> torch.Tensor:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        # Forward (from target to eye) in world axes
        dir_world = torch.tensor([sy * cp, sp, cy * cp], dtype=torch.float32, device=self.device)
        eye = self.target + self.radius * dir_world
        return eye

    def _rebuild_camera(self):
        eye = self._spherical_to_eye()
        self.cam = build_minicam(
            width=self.cam.image_width,
            height=self.cam.image_height,
            fov_deg=math.degrees(self.cam.FoVy),
            eye=eye,
            center=self.target,
            up=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
            device=self.device,
            znear=self.cam.znear,
            zfar=self.cam.zfar,
        )

    def _update_texture(self):
        img = render_view(self.cam, self.gaussians, self.pipeline, self.bg)
        _, _, rgba = tensor_to_rgba_list(img)
        dpg.set_value(self.texture_id, rgba)

    # ---------- mouse handlers ----------

    def on_mouse_down(self, button, x, y):
        self._last_mouse_pos = (x, y)
        if button == 0:   # LMB
            self._lmb_down = True
        elif button == 1: # RMB
            self._rmb_down = True

    def on_mouse_release(self, button, x, y):
        if button == 0:
            self._lmb_down = False
        elif button == 1:
            self._rmb_down = False

    def on_mouse_drag(self, button, data, dy):
        dx = data[1]
        dy = data[2]
        if button == 0 and self._lmb_down:
            print(f'dx: {dx}')
            print(f'dy: {dy}')
            # Orbit
            self.yaw   -= dx * self.rotate_sensitivity
            self.pitch -= dy * self.rotate_sensitivity
            # clamp pitch to avoid flip
            limit = math.radians(89.0)
            self.pitch = max(-limit, min(limit, self.pitch))
            self._rebuild_camera()
            self._update_texture()
        elif button == 1 and self._rmb_down:
            # Pan (move target along camera right/up)
            # camera basis from view matrix (rows are cam axes in world coords)
            w2v = self.cam.world_view_transform  # 4x4
            # world->view rows are [right; up; forward]; take first 3 comps; normalize not strictly needed
            right = w2v[0, :3]; up = w2v[1, :3]
            pan_scale = self.radius * self.pan_sensitivity
            self.target = (self.target
                           - right * (dx * pan_scale)
                           + up    * (dy * pan_scale))
            self._rebuild_camera()
            self._update_texture()

    def on_mouse_wheel(self, delta):
        # zoom: adjust radius multiplicatively
        # positive delta -> scroll up -> zoom in
        scale = math.exp(-self.zoom_sensitivity * float(delta))
        self.radius = max(1e-3, self.radius * scale)
        self._rebuild_camera()
        self._update_texture()

    # ---------- DPG setup & callbacks ----------

    def _initial_render(self):
        img = render_view(self.cam, self.gaussians, self.pipeline, self.bg)
        w, h, rgba = tensor_to_rgba_list(img)
        self.tex_width, self.tex_height = w, h
        return rgba

    def _attach_input_handlers(self):
        with dpg.handler_registry():
            # Mouse down / up
            dpg.add_mouse_click_handler(callback=lambda s, a, u: self.on_mouse_down(a, *dpg.get_mouse_pos()))
            dpg.add_mouse_release_handler(callback=lambda s, a, u: self.on_mouse_release(a, *dpg.get_mouse_pos()))
            # Drag (we need deltas)
            dpg.add_mouse_drag_handler(button=0, callback=lambda s, a, u: self.on_mouse_drag(0, a, a))  # dx=dy=a (DPG passes total?)
            dpg.add_mouse_drag_handler(button=1, callback=lambda s, a, u: self.on_mouse_drag(1, a, a))
            # The above generic drag signature isn't ideal in all DPG versions; fallback below using pos delta polling per frame.
            dpg.add_mouse_wheel_handler(callback=lambda s, a, u: self.on_mouse_wheel(a))

        # Fallback per-frame polling for robust dx/dy (works across DPG versions)
        def frame_update():
            x, y = dpg.get_mouse_pos()
            dx = x - self._last_mouse_pos[0]
            dy = y - self._last_mouse_pos[1]
            if (self._lmb_down or self._rmb_down) and (dx != 0 or dy != 0):
                if self._lmb_down:
                    self.on_mouse_drag(0, dx, dy)
                elif self._rmb_down:
                    self.on_mouse_drag(1, dx, dy)
            self._last_mouse_pos = (x, y)

        # Register the per-frame callback
        dpg.set_frame_callback(1, lambda: frame_update())

    def run(self):
        dpg.create_context()
        def save_init():
            dpg.save_init_file("dpg.ini")

        dpg.configure_app(init_file="dpg.ini")  # default file is 'dpg.ini'
        with dpg.window(label="about", tag="main window"):
            dpg.add_button(label="Save Window pos", callback=lambda: save_init)
        rgba = self._initial_render()

        dpg.create_viewport(
            title="GSIR Dear PyGui Viewer (Orbit Camera, No Scene)",
            width=max(800, self.tex_width + 200),
            height=max(600, self.tex_height + 200),
        )

        with dpg.texture_registry(show=False):
            self.texture_id = dpg.add_dynamic_texture(self.tex_width, self.tex_height, rgba)

        with dpg.window(label="GSIR Viewer", width=-1, height=-1, tag="Viewport"):
            with dpg.child_window(width=-1, height=-50, border=False):
                dpg.add_image(self.texture_id)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Re-render", callback=lambda: self._update_texture())

        with dpg.window(label="GSIR Controls", width=580, height=280):
            dpg.add_text("Model Selection")
            dpg.add_combo(
                label="Choose GSIR Model",
                items=["Model A", "Model B", "Model C"],
                default_value="Model A",
                callback=lambda s,a,u: print(f"Selected model: {a}")
            )
            dpg.add_separator()
            dpg.add_text("Environment Map")
            dpg.add_input_text(
                label="Env Map Path",
                hint="Path to HDR or EXR file",
                callback=lambda s,a,u: print(f"Env map path: {a}")
            )
            dpg.add_button(
                label="Load Environment Map",
                callback=lambda: print("Load env map clicked")
            )

        self._attach_input_handlers()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Viewport", True)
        dpg.start_dearpygui()
        dpg.destroy_context()


# ----------------------------- Main -----------------------------

def main():
    parser = ArgumentParser(description="GSIR Dear PyGui Viewer (Manual Camera, Orbit, No Scene)")

    # Keep pipeline-only args (renderer likely needs it)
    pipeline = PipelineParams(parser)

    # Essentials
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to GSIR gaussian checkpoint")
    parser.add_argument("--bg", type=float, default=0.0, help="Background gray value in [0,1]")
    parser.add_argument("--sh", type=int, default=3, help="SH degree for GaussianModel (match training)")

    # Camera + image settings
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fov_deg", type=float, default=60.0, help="Vertical FOV in degrees")
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)

    parser.add_argument("--eye", type=float, nargs=3, default=[0.0, 0.0, 3.0])
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, 1.0, 0.0])

    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Gaussians
    gaussians = load_gaussians_from_ckpt(args.checkpoint, sh_degree=args.sh, device=device)

    # Build initial MiniCam
    eye = torch.tensor(args.eye, dtype=torch.float32, device=device)
    center = torch.tensor(args.center, dtype=torch.float32, device=device)
    up = torch.tensor(args.up, dtype=torch.float32, device=device)
    cam = build_minicam(
        width=args.width,
        height=args.height,
        fov_deg=args.fov_deg,
        eye=eye,
        center=center,
        up=up,
        device=device,
        znear=args.znear,
        zfar=args.zfar,
    )

    app = GSIRViewerApp(cam=cam, gaussians=gaussians, pipeline=pipeline.extract(args), bg=args.bg)
    app.run()


if __name__ == "__main__":
    main()
