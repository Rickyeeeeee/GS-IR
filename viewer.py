# gsir_dearpygui_viewer.py
#
# Minimal Dear PyGui viewer that:
# 1) Loads a GSIR model/checkpoint
# 2) Renders a frame from a selected camera
# 3) Displays the RGB image in a Dear PyGui widget (with buttons to switch views/re-render)
#
# Requirements:
# - Your GS codebase in PYTHONPATH (arguments.py, gaussian_renderer.py, scene.py, etc.)
# - PyTorch + CUDA (optional but recommended)
# - dearpygui
#
# Example:
# python gsir_dearpygui_viewer.py \
#   -m output/garden-linear/ \
#   -s dataset/nerf_data/nerf_real_360/garden/ \
#   --checkpoint output/garden-linear/chkpnt35000.pth

import os
import sys
import torch
import numpy as np
import dearpygui.dearpygui as dpg

# ---- Import GSIR codebase pieces (same as in shadow_map.py) ----
from argparse import ArgumentParser
from typing import Dict, Tuple, Union

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene

# ----------------------------- Utils -----------------------------

def tensor_to_rgba_list(img: Union[torch.Tensor, np.ndarray]) -> Tuple[int, int, list]:
    """
    Convert a [3,H,W] torch float tensor in [0,1] (or np array [H,W,3]) to an RGBA float list
    that DearPyGui expects for a (dynamic/static) texture.
    Returns: (width, height, flat_list_rgba)
    """
    if isinstance(img, torch.Tensor):
        # [3,H,W] -> [H,W,3]
        img = img.detach().clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
    else:
        img = np.clip(img, 0.0, 1.0)

    h, w, _ = img.shape
    alpha = np.ones((h, w, 1), dtype=img.dtype)
    rgba = np.concatenate([img, alpha], axis=-1)  # [H,W,4] in [0,1]
    return w, h, rgba.reshape(-1).tolist()

# ---------------------------- Loader -----------------------------

@torch.no_grad()
def load_scene_and_gaussians(
    dataset: GroupParams,
    checkpoint_path: str,
) -> Tuple[Scene, GaussianModel]:
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    ckpt = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(ckpt, tuple):
        model_params = ckpt[0]
    elif isinstance(ckpt, dict):
        model_params = ckpt["gaussians"]
    else:
        raise TypeError("Unsupported checkpoint format.")

    gaussians.restore(model_params)
    return scene, gaussians

@torch.no_grad()
def render_view(
    scene: Scene,
    pipeline: GroupParams,
    view_index: int = 0,
    bg_color_val: float = 0.0,
) -> torch.Tensor:
    """
    Renders and returns a [3,H,W] float tensor in [0,1].
    """
    views = scene.getTrainCameras()
    view_index = max(0, min(view_index, len(views) - 1))
    cam = views[view_index]
    cam.bg_color[...] = bg_color_val

    background = cam.bg_color.cuda() if torch.cuda.is_available() else cam.bg_color
    out: Dict[str, torch.Tensor] = render(
        viewpoint_camera=cam,
        pc=scene.gaussians,
        pipe=pipeline,
        bg_color=background,
        inference=True,
        derive_normal=False,
    )
    # out["render"] is [3,H,W] in [0,1]
    return out["render"]

# ----------------------------- GUI ------------------------------

class GSIRViewerApp:
    def __init__(self, scene: Scene, pipeline: GroupParams):
        self.scene = scene
        self.pipeline_params = pipeline
        self.views = self.scene.getTrainCameras()
        self.view_idx = 0
        self.texture_id = None
        self.tex_width = 0
        self.tex_height = 0

    def _initial_render(self):
        img = render_view(self.scene, self.pipeline_params, self.view_idx)
        w, h, rgba = tensor_to_rgba_list(img)
        self.tex_width, self.tex_height = w, h
        return rgba

    def _update_texture(self):
        img = render_view(self.scene, self.pipeline_params, self.view_idx)
        _, _, rgba = tensor_to_rgba_list(img)
        dpg.set_value(self.texture_id, rgba)

    def next_view(self, _sender, _app_data, _user_data):
        self.view_idx = (self.view_idx + 1) % len(self.views)
        self._update_texture()

    def prev_view(self, _sender, _app_data, _user_data):
        self.view_idx = (self.view_idx - 1) % len(self.views)
        self._update_texture()

    def rerender(self, _s, _a, _u):
        self._update_texture()

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="GSIR Dear PyGui Viewer", width=max(800, self.tex_width+200), height=max(600, self.tex_height+200))

        with dpg.texture_registry(show=True):
            rgba = self._initial_render()
            self.texture_id = dpg.add_dynamic_texture(self.tex_width, self.tex_height, rgba)

        with dpg.window(label="GSIR Viewer", width=-1, height=-1):
            dpg.add_text(f"Views: {len(self.views)}")
            with dpg.group(horizontal=True):
                dpg.add_button(label="⟨ Prev", callback=self.prev_view)
                dpg.add_button(label="Re-render", callback=self.rerender)
                dpg.add_button(label="Next ⟩", callback=self.next_view)
            dpg.add_spacing(count=1)
            dpg.add_separator()
            dpg.add_spacing(count=1)
            # The image draws at native resolution; wrap in child for scroll if needed
            with dpg.child_window(width=-1, height=-1, border=False):
                dpg.add_image(self.texture_id)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

# ----------------------------- Main -----------------------------

def main():
    # Parse args like shadow_map.py so it integrates with your pipeline
    parser = ArgumentParser(description="GSIR Dear PyGui Viewer")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--bg", type=float, default=0.0, help="Background gray value in [0,1]")
    args = get_combined_args(parser)
    args.eval = False

    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Load scene & gaussians
    scene, _ = load_scene_and_gaussians(dataset=model.extract(args), checkpoint_path=args.checkpoint)

    # Run the GUI app
    app = GSIRViewerApp(scene=scene, pipeline=pipeline.extract(args))
    app.run()

if __name__ == "__main__":
    main()
