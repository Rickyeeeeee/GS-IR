# viewer.py
#
# Dear PyGui viewer for GS-IR relighting with PBR shading.
#
# Requirements (available in your GS-IR repo):
#   arguments.py, gaussian_renderer.py, pbr.py, scene.py, utils.*
# Plus: dearpygui, nvdiffrast.torch, torchvision, torch, numpy, opencv
#

import os
import math
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional
import time
import json

import numpy as np
import torch
import torch.nn.functional as F
import dearpygui.dearpygui as dpg

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene import Scene
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import build_rotation, rotation_to_quaternion, safe_state
from utils.viewer_utils import get_canonical_rays, tensor_to_raw_rgba, euler_to_matrix, _aces_film, _sample_env_latlong, latlong_to_cubemap, read_hdr, _linear_to_srgb
from viewer_camera import ViewerCamera

# -----------------------------
# HDRI PRESETS 
# -----------------------------
# Note: These are now just filenames. Provide the root path via the --hdri_root argument.
HDRI_PRESETS: List[Tuple[str, str]] = [
    ("Bridge", "bridge.hdr"),
    ("City", "city.hdr"),
    ("Courtyard", "courtyard.hdr"),
    ("Fireplace", "fireplace.hdr"),
    ("Forest", "forest.hdr"),
    ("Interior", "interior.hdr"),
    ("Museum", "museum.hdr"),
    ("Night", "night.hdr"),
    ("Snow", "snow.hdr"),
    ("Square", "square.hdr"),
    ("Studio", "studio.hdr"),
    ("Sunrise", "sunrise.hdr"),
    ("Sunset", "sunset.hdr"),
    ("Tunnel", "tunnel.hdr"),
]

# -----------------------------
# Viewer App
# -----------------------------

class RelightViewer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- NEW: Load multiple GS-IR models ---
        self.models: List[GaussianModel] = []
        self.model_names: List[str] = []
        for i, ckpt_path in enumerate(args.checkpoint):
            model_name = f"Model {i+1} ({os.path.basename(ckpt_path)})"
            print(f"[viewer] Loading: {model_name}")
            gaussians = GaussianModel(args.sh_degree)
            self._load_checkpoint(gaussians, ckpt_path)
            self.models.append(gaussians)
            self.model_names.append(model_name)
        
        # This is the single model used for rendering after concatenation
        self.render_gaussians = GaussianModel(args.sh_degree)
        
        # --- NEW: Per-model transform states ---
        self.model_transforms = [
            {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0} for _ in self.models
        ]
        self.active_model_idx = 0

        # Base Gaussian Rotation Fix (the original hardcoded R_fix)
        self.base_R_fix = torch.from_numpy(euler_to_matrix(
            torch.deg2rad(torch.tensor(0.0)), 
            torch.deg2rad(torch.tensor(90.0)), 
            torch.deg2rad(torch.tensor(0.0)))).cuda()

        # HDRI presets & cache
        full_path_presets = []
        if args.hdri_root and os.path.isdir(args.hdri_root):
            print(f"[viewer] Loading HDRI presets from: {args.hdri_root}")
            for label, filename in HDRI_PRESETS:
                full_path_presets.append((label, os.path.join(args.hdri_root, filename)))
        
        self.hdri_presets: List[Tuple[str, str]] = full_path_presets

        if args.hdri is not None:
            if not any(os.path.normpath(p) == os.path.normpath(args.hdri) for _, p in self.hdri_presets):
                self.hdri_presets.insert(0, ("(from --hdri)", args.hdri))
        
        if not self.hdri_presets:
            raise ValueError("No HDRIs found. Please provide a valid --hdri_root directory or a specific --hdri file.")

        self.hdri_labels = [lbl for (lbl, _) in self.hdri_presets]
        self.hdri_paths = {lbl: path for (lbl, path) in self.hdri_presets}
        self.hdri_label_current = self._label_from_path(args.hdri) if args.hdri else self.hdri_labels[0]
        
        self.hdri_cache_latlong: Dict[str, torch.Tensor] = {} 
        self.hdri_cache_cubemap: Dict[str, torch.Tensor] = {} 

        # Light & BRDF
        self.light = CubemapLight(base_res=args.env_res).to(self.device)
        self.brdf_lut = get_brdf_lut().to(self.device)
        self._ensure_hdri(self.hdri_label_current)
        self.light.eval()

        # Camera setup
        self.camera = ViewerCamera(
            FoVx=1.6609910500147054, FoVy=1.103431263015383, W=959, H=539, data_device="cuda"
        )
        # Target the center of the first model
        self.target = self.models[0].get_xyz.mean(dim=0).detach().cpu().numpy()
        self.camera.look_at(self.target, distance=1.0)
        
        # Render state & UI controls
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        self.enable_tone = args.tone
        self.enable_gamma = args.gamma
        self.enable_metallic = args.metallic
        self.show_env_bg = not getattr(args, "no_env_bg", False)
        
        self.hdri_rotation_deg = 0.0 # Yaw for the environment map

        # DearPyGui state
        self.texture_id = None
        self.tex_size = (0, 0)
        self.last_mouse_pos = None

    # ----- HDRI helpers -----
    def _label_from_path(self, path: str) -> str:
        if path is None: return None
        norm = os.path.normpath(path)
        for lbl, p in self.hdri_presets:
            if os.path.normpath(p) == norm:
                return lbl
        return self.hdri_labels[0]

    def _ensure_hdri(self, label: str):
        path = self.hdri_paths[label]
        if label not in self.hdri_cache_latlong:
            hdri_np = read_hdr(path)
            self.hdri_cache_latlong[label] = torch.from_numpy(hdri_np).to(self.device)
        latlong = self.hdri_cache_latlong[label]

        if label not in self.hdri_cache_cubemap:
            self.hdri_cache_cubemap[label] = latlong_to_cubemap(latlong, [self.args.env_res, self.args.env_res])

        self.hdri = self.hdri_cache_latlong[label]
        self.light.base.data = self.hdri_cache_cubemap[label]
        self.light.build_mips()

    def _load_checkpoint(self, gaussians: GaussianModel, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        model_params = checkpoint.get("gaussians", checkpoint) if isinstance(checkpoint, Dict) else checkpoint[0]
        gaussians.restore(model_params)

    # ---------------------------------------------------
    # NEW: Function to concatenate Gaussians before render
    # ---------------------------------------------------
    def _concatenate_gaussians(self):
        """
        Apply individual transforms and concatenate all models into a single
        GaussianModel for rendering.
        
        <<< IMPLEMENT YOUR CONCATENATION LOGIC HERE >>>
        """
        # Example Implementation (replace with your own logic):
        # This is a basic example. You might need more sophisticated handling of
        # all the different tensor attributes of the GaussianModel.
        
        all_xyz = []
        all_normals = []
        all_rotations = []
        # ... gather all other attributes like features, scaling, opacity ...

        for i, model in enumerate(self.models):
            trans = self.model_transforms[i]
            
            # Get the original, unmodified data for this model
            xyz = model.get_xyz.clone().detach()
            normal = model.get_normal.clone().detach()
            rotation = model.get_rotation.clone().detach()

            # 1. Calculate user-defined rotation for this model
            R_user = torch.from_numpy(euler_to_matrix(
                torch.deg2rad(torch.tensor(trans['yaw'])), 
                torch.deg2rad(torch.tensor(trans['pitch'])), 
                torch.deg2rad(torch.tensor(trans['roll'])))).cuda()
        
            # 2. Combine with the original fixed rotation
            R_combined = R_user @ self.base_R_fix[:3, :3]
        
            # 3. Apply the combined rotation
            transformed_xyz = (R_combined @ xyz.T).T
            transformed_normal = (R_combined @ normal.T).T
            transformed_rotation = rotation_to_quaternion(R_combined @ build_rotation(rotation))

            # 4. Append transformed tensors to lists
            all_xyz.append(transformed_xyz)
            all_normals.append(transformed_normal)
            all_rotations.append(transformed_rotation)
            # ... append other attributes ...

        # After the loop, concatenate all lists of tensors
        # self.render_gaussians._xyz = torch.cat(all_xyz, dim=0)
        # self.render_gaussians._normal = torch.cat(all_normals, dim=0)
        # self.render_gaussians._rotation = torch.cat(all_rotations, dim=0)
        # ... and so on for all other attributes ...

        # For now, as a placeholder, we just render the active model
        # Replace this with your full concatenated model
        active_model = self.models[self.active_model_idx]
        trans = self.model_transforms[self.active_model_idx]
        R_user = torch.from_numpy(euler_to_matrix(
            torch.deg2rad(torch.tensor(trans['yaw'])), 
            torch.deg2rad(torch.tensor(trans['pitch'])), 
            torch.deg2rad(torch.tensor(trans['roll'])))).cuda()
        R_combined = R_user @ self.base_R_fix[:3, :3]
        
        self.render_gaussians._xyz = (R_combined @ active_model.get_xyz.T).T
        self.render_gaussians._normal = (R_combined @ active_model.get_normal.T).T
        self.render_gaussians._rotation = rotation_to_quaternion(R_combined @ build_rotation(active_model.get_rotation))
        self.render_gaussians._features_dc = active_model._features_dc
        self.render_gaussians._features_rest = active_model._features_rest
        self.render_gaussians._scaling = active_model._scaling
        self.render_gaussians._opacity = active_model._opacity
        self.render_gaussians._albedo = active_model._albedo
        self.render_gaussians._roughness = active_model._roughness
        self.render_gaussians._metallic = active_model._metallic


    # -------------------------
    # Rendering
    # -------------------------
    @torch.no_grad()
    def render_current(self) -> torch.Tensor:
        
        # --- NEW: Prepare the combined model for rendering ---
        self._concatenate_gaussians()
        
        view = self.camera

        # GS-IR forward pass
        rendering_result = render(
            viewpoint_camera=view, pc=self.render_gaussians, pipe=self.args.pipeline, 
            bg_color=self.background, inference=True, pad_normal=True, derive_normal=True,
        )

        normal_map = rendering_result["normal_map"]
        opacity_mask = rendering_result["opacity_map"]
        albedo_map = rendering_result["albedo_map"]
        roughness_map = rendering_result["roughness_map"]
        metallic_map = rendering_result["metallic_map"]

        # World-space View directions
        H, W = view.image_height, view.image_width
        c2w = torch.inverse(view.world_view_transform.T)
        canonical_rays = get_canonical_rays(H, W, self.camera.FoVx, self.camera.FoVy)
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3]).sum(dim=-1).reshape(H, W, 3)
        )  # [H,W,3]

        # --- HDRI Environment Rotation (Yaw) ---
        env_yaw_rad = math.radians(self.hdri_rotation_deg)
        env_cos_yaw = math.cos(env_yaw_rad)
        env_sin_yaw = math.sin(env_yaw_rad)
        
        x, y, z = view_dirs[..., 0], view_dirs[..., 1], view_dirs[..., 2]
        
        # R_y(yaw) on view directions
        x_rotated = x * env_cos_yaw - z * env_sin_yaw
        z_rotated = x * env_sin_yaw + z * env_cos_yaw
        light_sample_dirs = torch.stack([x_rotated, y, z_rotated], dim=-1) # [H,W,3]
        # ---------------------------------------

        # PBR shading (uses ROTATED direction vector for environment map sampling)
        result = pbr_shading(
            light=self.light,
            normals=normal_map.permute(1, 2, 0),
            view_dirs=light_sample_dirs,        
            mask=rendering_result["normal_mask"].permute(1, 2, 0),
            albedo=albedo_map.permute(1, 2, 0),
            roughness=roughness_map.permute(1, 2, 0),
            metallic=metallic_map.permute(1, 2, 0) if self.enable_metallic else None,
            tone=self.enable_tone,
            gamma=self.enable_gamma,
            brdf_lut=self.brdf_lut,
        )
        render_rgb = result["render_rgb"].clamp(0.0, 1.0)

        # Composite the HDRI as a background (uses ROTATED direction vector)
        if self.show_env_bg:
            env_rgb = _sample_env_latlong(self.hdri, light_sample_dirs) 
            if self.enable_tone:
                env_rgb = _aces_film(env_rgb)
            if self.enable_gamma:
                env_rgb = _linear_to_srgb(env_rgb)
            bg_mask = 1.0 - opacity_mask.permute(1, 2, 0).clamp(0.0, 1.0)
            render_rgb = render_rgb * (1.0 - bg_mask) + env_rgb * bg_mask

        return render_rgb.clamp(0.0, 1.0)

    # -------------------------
    # DearPyGui UI
    # -------------------------
    def _update_image(self, img_rgb: torch.Tensor):
        H, W, _ = img_rgb.shape
        rgba_np = tensor_to_raw_rgba(img_rgb)
        
        if self.texture_id is None or self.tex_size != (W, H):
            try:
                if self.texture_id is not None: dpg.delete_item(self.texture_id)
            except Exception: pass
            with dpg.texture_registry(show=False):
                self.texture_id = dpg.add_raw_texture(W, H, rgba_np, format=dpg.mvFormat_Float_rgba)
            self.tex_size = (W, H)
            if dpg.does_item_exist("render_image"):
                dpg.configure_item("render_image", texture_tag=self.texture_id)
        else:
            dpg.set_value(self.texture_id, rgba_np)

    def run(self):
        dpg.create_context()
        img = self.render_current()
        H, W, _ = img.shape

        dpg.create_viewport(title="GS-IR PBR Viewer", width=self.args.width, height=self.args.height)

        with dpg.texture_registry(show=False):
            self.texture_id = dpg.add_raw_texture(W, H, tensor_to_raw_rgba(img), format=dpg.mvFormat_Float_rgba)

        # --- Helpers ---
        def _rebuild_projection_for(camera, w: int, h: int):
            if w <= 0 or h <= 0: return
            aspect = float(w) / float(h)
            tan_half_y = float(camera.FoVy)
            tan_half_x = tan_half_y * aspect
            camera.FoVx, camera.FoVy = tan_half_x, tan_half_y
            camera.projection_matrix = (getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx, fovY=camera.FoVy).transpose(0, 1).to(camera.data_device))
            camera.update_matrices()
            camera.image = torch.zeros((3, h, w), dtype=torch.float32)

        def render_callback(sender=None, app_data=None):
            try:
                if torch.cuda.is_available(): torch.cuda.synchronize(self.device)
                img2 = self.render_current()
                self._update_image(img2)
            except Exception as e:
                print(f"[viewer] Render error: {e}")

        def _apply_resize(new_w: int, new_h: int):
            if new_w <= 0 or new_h <= 0: return
            _rebuild_projection_for(self.camera, new_w, new_h)
            if dpg.does_item_exist("render_image"):
                dpg.configure_item("render_image", width=new_w - 20, height=new_h - 20)
            render_callback()

        # --- UI Callbacks ---
        def on_toggle_tone(sender, app_data): self.enable_tone = bool(app_data); render_callback()
        def on_toggle_gamma(sender, app_data): self.enable_gamma = bool(app_data); render_callback()
        def on_toggle_env(sender, app_data): self.show_env_bg = bool(app_data); render_callback()
        def on_hdri_rotation_change(sender, app_data): self.hdri_rotation_deg = app_data; render_callback()
        
        # --- NEW: Callbacks for model transforms ---
        def on_active_model_change(sender, app_data):
            # Find the index of the selected model name
            self.active_model_idx = self.model_names.index(app_data)
            # Update sliders to reflect the newly selected model's transforms
            trans = self.model_transforms[self.active_model_idx]
            dpg.set_value("model_yaw", trans['yaw'])
            dpg.set_value("model_pitch", trans['pitch'])
            dpg.set_value("model_roll", trans['roll'])
            render_callback()

        def on_model_yaw_change(sender, app_data):
            self.model_transforms[self.active_model_idx]['yaw'] = app_data
            render_callback()
        
        def on_model_pitch_change(sender, app_data):
            self.model_transforms[self.active_model_idx]['pitch'] = app_data
            render_callback()

        def on_model_roll_change(sender, app_data):
            self.model_transforms[self.active_model_idx]['roll'] = app_data
            render_callback()

        def on_hdri_change(sender, app_data):
            self.hdri_label_current = app_data
            try:
                self._ensure_hdri(self.hdri_label_current)
            except Exception as e:
                print(f"[viewer] HDRI switch error: {e}")
            render_callback()

        # --- Windows ---
        with dpg.window(label="Controls", width=380, height=-1, pos=(10, 10)):
            dpg.add_text("FPS: --", tag="fps_display")
            dpg.add_text("Frame Time: -- ms", tag="frametime_display")
            
            dpg.add_separator()
            dpg.add_text("Model Controls")
            dpg.add_combo(items=self.model_names, default_value=self.model_names[0], label="Active Model", callback=on_active_model_change)
            dpg.add_slider_float(label="Model Yaw", tag="model_yaw", min_value=-180.0, max_value=180.0, default_value=0.0, format="%.1f deg", callback=on_model_yaw_change)
            dpg.add_slider_float(label="Model Pitch", tag="model_pitch", min_value=-180.0, max_value=180.0, default_value=0.0, format="%.1f deg", callback=on_model_pitch_change)
            dpg.add_slider_float(label="Model Roll", tag="model_roll", min_value=-180.0, max_value=180.0, default_value=0.0, format="%.1f deg", callback=on_model_roll_change)
            
            dpg.add_separator()
            dpg.add_text("Environment")
            dpg.add_combo(items=self.hdri_labels, default_value=self.hdri_label_current, label="HDRI Preset", callback=on_hdri_change)
            dpg.add_slider_float(label="HDRI Yaw", min_value=-180.0, max_value=180.0, default_value=self.hdri_rotation_deg, format="%.1f deg", callback=on_hdri_rotation_change)

            dpg.add_separator()
            dpg.add_text("Shading")
            dpg.add_checkbox(label="ACES tone mapping", default_value=self.enable_tone, callback=on_toggle_tone)
            dpg.add_checkbox(label="Gamma correction (sRGB)", default_value=self.enable_gamma, callback=on_toggle_gamma)
            dpg.add_checkbox(label="HDRI as background", default_value=self.show_env_bg, callback=on_toggle_env)

            dpg.add_separator()
            dpg.add_button(label="Render Once", callback=render_callback)

            dpg.add_separator()
            checkpoints_str = "\n".join(self.args.checkpoint)
            dpg.add_text(f"Checkpoints:\n{checkpoints_str}")

        with dpg.window(label="Render", tag="Render", width=self.args.width - 410, height=self.args.height - 40, pos=(400, 10)):
            dpg.add_image(texture_tag=self.texture_id, tag="render_image", width=W, height=H)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Render", True)
        def _on_viewport_resize(sender, app_data, user_data): _apply_resize(dpg.get_viewport_client_width(), dpg.get_viewport_client_height())
        dpg.set_viewport_resize_callback(callback=_on_viewport_resize)
        _on_viewport_resize(None, None, None)

        # --- Input loop ---
        MOVE_SPEED, ORBIT_KEY_SPEED, MOUSE_SENSITIVITY, MAX_DT = 4.0, 4.0, 0.002, 0.10
        _last_time = time.perf_counter()

        while dpg.is_dearpygui_running():
            now = time.perf_counter()
            dt = min(now - _last_time, MAX_DT)
            _last_time = now
            
            if dt > 0.0:
                dpg.set_value("fps_display", f"FPS: {1.0 / dt:.1f}")
                dpg.set_value("frametime_display", f"Frame Time: {dt * 1000.0:.2f} ms")

            # Camera movement
            if dpg.is_key_down(dpg.mvKey_W): self.camera.move_forward(+MOVE_SPEED * dt)
            if dpg.is_key_down(dpg.mvKey_S): self.camera.move_forward(-MOVE_SPEED * dt)
            if dpg.is_key_down(dpg.mvKey_E): self.camera.move_up(-MOVE_SPEED * dt)
            if dpg.is_key_down(dpg.mvKey_Q): self.camera.move_up(+MOVE_SPEED * dt)
            if dpg.is_key_down(dpg.mvKey_D): self.camera.move_right(+MOVE_SPEED * dt)
            if dpg.is_key_down(dpg.mvKey_A): self.camera.move_right(-MOVE_SPEED * dt)

            # Camera orbit
            yaw_delta, pitch_delta = 0.0, 0.0
            if dpg.is_key_down(dpg.mvKey_Left):  yaw_delta  += +ORBIT_KEY_SPEED * dt
            if dpg.is_key_down(dpg.mvKey_Right): yaw_delta  += -ORBIT_KEY_SPEED * dt
            if dpg.is_key_down(dpg.mvKey_Up):    pitch_delta += -ORBIT_KEY_SPEED * dt
            if dpg.is_key_down(dpg.mvKey_Down):  pitch_delta += +ORBIT_KEY_SPEED * dt
            if yaw_delta or pitch_delta: self.camera.orbit(yaw_delta, pitch_delta)

            if dpg.is_mouse_button_dragging(dpg.mvMouseButton_Left, threshold=0.0):
                current_pos = dpg.get_mouse_pos(local=False)
                if self.last_mouse_pos is None: self.last_mouse_pos = current_pos
                dx = current_pos[0] - self.last_mouse_pos[0]
                dy = current_pos[1] - self.last_mouse_pos[1]
                self.last_mouse_pos = current_pos
                self.camera.orbit(-dx * MOUSE_SENSITIVITY, +dy * MOUSE_SENSITIVITY)
            else: self.last_mouse_pos = None

            render_callback()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="GS-IR PBR DearPyGui Viewer")

    # Define all arguments first
    parser.add_argument("--config", type=str, default="config.json", help="Path to a JSON configuration file.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    # NEW: --checkpoint can now accept multiple values
    parser.add_argument("--checkpoint", type=str, default=None, nargs='+', help="Path(s) to GS-IR checkpoint(s) to load.")
    parser.add_argument("--hdri", type=str, default=None, help="Path to a specific latlong HDRI file (.hdr). Overrides presets.")
    parser.add_argument("--hdri_root", type=str, default=None, help="Root directory containing HDRI files for presets.")
    parser.add_argument("--env_res", type=int, default=256, help="Cubemap base resolution per face.")
    parser.add_argument("--width", type=int, default=1280, help="Viewer window width.")
    parser.add_argument("--height", type=int, default=720, help="Viewer window height.")
    parser.add_argument("--tone", action="store_true", help="Enable ACES filmic tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear->sRGB gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Use reconstructed metallic map.")
    parser.add_argument("--no_env_bg", action="store_true", help="Disable compositing HDRI as background.")

    # Temporarily parse for config path
    temp_args, _ = parser.parse_known_args()

    # Load defaults from config file if it exists, AFTER arguments are defined
    if os.path.isfile(temp_args.config):
        print(f"[viewer] Loading arguments from: {temp_args.config}")
        with open(temp_args.config, 'r') as f:
            config_data = json.load(f)
        parser.set_defaults(**config_data)
    else:
        print(f"[viewer] Config file not found at '{temp_args.config}'. Using command-line arguments and defaults.")

    # Now, parse all arguments with the correct defaults
    args = parser.parse_args()

    # --- Argument Validation ---
    if not args.checkpoint:
        parser.error("A --checkpoint must be provided either via command-line or config file.")

    if not args.hdri and not args.hdri_root:
        parser.error("An HDRI source must be provided via --hdri (specific file) or --hdri_root (presets directory).")

    # Compatibility structure
    class _Args:
        pass
    _a = _Args()
    _a.__dict__.update(vars(args))
    _a.pipeline = pipeline.extract(args)
    _a.sh_degree = args.sh_degree

    safe_state(getattr(args, "quiet", False))

    app = RelightViewer(_a)
    app.run()