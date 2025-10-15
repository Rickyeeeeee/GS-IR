"""
ImGui Bundle based viewer for GS-IR relighting with PBR shading.

This module ports the DearPyGui viewer to the imgui_bundle 1.3.0 toolchain.
"""

from __future__ import annotations

import math
import os
import time
import json
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import build_rotation, rotation_to_quaternion, safe_state
from utils.viewer_utils import (
    _aces_film,
    _linear_to_srgb,
    _sample_env_latlong,
    euler_to_matrix,
    get_canonical_rays,
    latlong_to_cubemap,
    read_hdr,
    tensor_to_raw_rgba,
)
from viewer_camera import ViewerCamera

try:
    from imgui_bundle import imgui, immapp, hello_imgui, immvision
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "imgui_bundle 1.3.0 (with immvision) is required for imgui_bundle_viewer.py"
    ) from exc


# -----------------------------
# HDRI PRESETS
# -----------------------------
# Note: These are filenames. Provide the root path via the --hdri_root argument.
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


def _name_variants(name: str) -> List[str]:
    if not name:
        return []
    variants = {name}
    variants.add(name.lower())
    variants.add(name.upper())
    snake = []
    current = []
    for idx, ch in enumerate(name):
        if ch.isupper() and idx > 0 and (not name[idx - 1].isupper()):
            current.append("_")
        current.append(ch.lower())
    snake_name = "".join(current)
    if snake_name:
        snake.append(snake_name)
    camel_alt = name.replace(" ", "_").replace("-", "_")
    variants.add(camel_alt)
    variants.add(camel_alt.lower())
    variants.add(camel_alt.upper())
    if snake:
        variants.update(snake)
    return [v for v in variants if v]


def _try_resolve_imgui_constant(family: str, *names: str):
    """Resolve an ImGui enum member in a version tolerant way."""
    containers_map = {
        "key": ["Key", "ImGuiKey"],
        "mouse_button": ["MouseButton", "MouseButton_", "ImGuiMouseButton"],
        "cond": ["Cond", "Cond_", "ImGuiCond"],
        "hovered": ["HoveredFlags", "HoveredFlags_", "ImGuiHoveredFlags"],
    }
    containers = containers_map.get(family, [])

    for container_name in containers:
        container = getattr(imgui, container_name, None)
        if container is None:
            continue
        members = getattr(container, "__members__", None)
        for original in names:
            for variant in _name_variants(original):
                if members and variant in members:
                    return members[variant]
                value = getattr(container, variant, None)
                if value is not None:
                    return value

    for name in names:
        for variant in _name_variants(name):
            value = getattr(imgui, variant, None)
            if value is not None:
                return value

    return None


def _imgui_key(name: str):
    return _try_resolve_imgui_constant(
        "key",
        name,
        name.upper(),
        name.lower(),
        f"Key_{name}",
        f"key_{name}",
    )


def _imgui_cond(name: str):
    return _try_resolve_imgui_constant(
        "cond",
        name,
        name.capitalize(),
        name.lower(),
    )


def _imgui_mouse_button(name: str):
    return _try_resolve_imgui_constant(
        "mouse_button",
        name,
        name.capitalize(),
        name.lower(),
    )


def _imgui_hovered_flag(name: str, default: int = 0):
    value = _try_resolve_imgui_constant(
        "hovered",
        name,
        name.capitalize(),
        name.lower(),
    )
    return int(value) if value is not None else default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class RelightViewer:
    """Viewer application using imgui_bundle."""

    MOVE_SPEED = 4.0
    ORBIT_KEY_SPEED = 4.0
    MOUSE_SENSITIVITY = 0.002
    MAX_DT = 0.10

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Load GS-IR models ---
        self.models: List[GaussianModel] = []
        self.model_names: List[str] = []
        for i, ckpt_path in enumerate(args.checkpoint):
            model_name = f"Model {i + 1} ({os.path.basename(ckpt_path)})"
            print(f"[viewer] Loading: {model_name}")
            gaussians = GaussianModel(args.sh_degree)
            self._load_checkpoint(gaussians, ckpt_path)
            self.models.append(gaussians)
            self.model_names.append(model_name)

        # Instance used for rendering (individual or concatenated)
        self.render_gaussians = GaussianModel(args.sh_degree)

        # Per-model transforms
        self.model_transforms = [
            {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "translation": [0.0, 0.0, 0.0],
                "scale": 1.0,
                "bbox_min": [-1e6, -1e6, -1e6],
                "bbox_max": [1e6, 1e6, 1e6],
            }
            for _ in self.models
        ]
        self.active_model_idx = 0
        self.render_jointly = False
        self.transform_state_path = getattr(args, "transform_state", None)
        if self.transform_state_path:
            self.transform_state_path = os.path.abspath(self.transform_state_path)
        self._transform_state_cache: Dict[str, Dict[str, object]] = {}
        self.model_render_cache: List[Dict[str, object]] = [
            {"attrs": None, "dirty": True} for _ in self.models
        ]
        self.joint_render_cache: Dict[str, object] = {"attrs": None, "dirty": True}
        self.profile_timings: Dict[str, float] = {}
        self._load_transform_state()

        # Base Gaussian Rotation Fix (the original hardcoded R_fix)
        self.base_R_fix = torch.from_numpy(
            euler_to_matrix(
                torch.deg2rad(torch.tensor(0.0)),
                torch.deg2rad(torch.tensor(90.0)),
                torch.deg2rad(torch.tensor(0.0)),
            )
        ).cuda()

        # HDRI presets
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
            raise ValueError(
                "No HDRIs found. Provide --hdri_root directory or a specific --hdri file."
            )

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
            FoVx=1.6609910500147054, FoVy=1.103431263015383, W=args.width - 320, H=args.height - 40, data_device="cuda"
        )
        self.target = self.models[0].get_xyz.mean(dim=0).detach().cpu().numpy()
        self.camera.look_at(self.target, distance=1.0)

        # Render state & controls
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        self.enable_tone = args.tone
        self.enable_gamma = args.gamma
        self.enable_metallic = args.metallic
        self.show_env_bg = not getattr(args, "no_env_bg", False)

        self.hdri_rotation_deg = 0.0

        # imgui state
        self.last_render_error: Optional[str] = None
        self.render_image_rgb: Optional[np.ndarray] = None
        self.render_image_rgba: Optional[np.ndarray] = None
        self._force_rerender = True
        self._last_render_time = 0.0
        self._last_frame_dt = 0.0
        self._total_frames = 0
        self._render_window_hovered = False
        self._render_window_focused = False
        self._render_image_hovered = False
        self._render_image_active = False
        self._mouse_dragging = False

        self._key_cache = {
            "forward": _imgui_key("W"),
            "back": _imgui_key("S"),
            "left": _imgui_key("A"),
            "right": _imgui_key("D"),
            "up": _imgui_key("Q"),
            "down": _imgui_key("E"),
            "yaw_left": _imgui_key("LeftArrow"),
            "yaw_right": _imgui_key("RightArrow"),
            "pitch_up": _imgui_key("UpArrow"),
            "pitch_down": _imgui_key("DownArrow"),
        }
        self._key_names = {
            "forward": "W",
            "back": "S",
            "left": "A",
            "right": "D",
            "up": "Q",
            "down": "E",
            "yaw_left": "LeftArrow",
            "yaw_right": "RightArrow",
            "pitch_up": "UpArrow",
            "pitch_down": "DownArrow",
        }
        self._key_state = {name: False for name in self._key_cache}
        self._mouse_left = _imgui_mouse_button("Left")
        if self._mouse_left is None:
            self._mouse_left = 0

        # Prime the render buffer so the first frame displays immediately.
        self._update_render_buffer(force=True)

    # ----- HDRI helpers -----
    def _label_from_path(self, path: str) -> Optional[str]:
        if path is None:
            return None
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
            self.hdri_cache_cubemap[label] = latlong_to_cubemap(
                latlong, [self.args.env_res, self.args.env_res]
            )

        self.hdri = self.hdri_cache_latlong[label]
        self.light.base.data = self.hdri_cache_cubemap[label]
        self.light.build_mips()

    def _load_checkpoint(self, gaussians: GaussianModel, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        model_params = checkpoint.get("gaussians", checkpoint) if isinstance(checkpoint, Dict) else checkpoint[0]
        gaussians.restore(model_params)

    def _state_key_for_checkpoint(self, ckpt_path: str) -> str:
        abs_path = os.path.abspath(ckpt_path)
        scene_dir = os.path.basename(os.path.dirname(abs_path))
        if not scene_dir:
            scene_dir = os.path.splitext(os.path.basename(abs_path))[0]
        return scene_dir

    def _load_transform_state(self):
        if not self.transform_state_path or not os.path.isfile(self.transform_state_path):
            return
        try:
            with open(self.transform_state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[viewer] Failed to load transform state: {exc}")
            return

        checkpoint_data = payload.get("checkpoints", {})
        self._transform_state_cache = checkpoint_data

        for idx, ckpt_path in enumerate(self.args.checkpoint):
            key = self._state_key_for_checkpoint(ckpt_path)
            saved = checkpoint_data.get(key)
            if saved is None:
                legacy_key = os.path.abspath(ckpt_path)
                saved = checkpoint_data.get(legacy_key)
            if not isinstance(saved, dict):
                continue
            trans = self.model_transforms[idx]
            trans["yaw"] = float(saved.get("yaw", trans["yaw"]))
            trans["pitch"] = float(saved.get("pitch", trans["pitch"]))
            trans["roll"] = float(saved.get("roll", trans["roll"]))
            saved_translation = saved.get("translation", trans["translation"])
            if isinstance(saved_translation, (list, tuple)) and len(saved_translation) == 3:
                trans["translation"] = [float(v) for v in saved_translation]
            trans["scale"] = max(1e-6, float(saved.get("scale", trans["scale"])))
            saved_bbox_min = saved.get("bbox_min")
            if isinstance(saved_bbox_min, (list, tuple)) and len(saved_bbox_min) == 3:
                trans["bbox_min"] = [float(v) for v in saved_bbox_min]
            saved_bbox_max = saved.get("bbox_max")
            if isinstance(saved_bbox_max, (list, tuple)) and len(saved_bbox_max) == 3:
                trans["bbox_max"] = [float(v) for v in saved_bbox_max]
            self._ensure_bbox_consistency(trans)
            self._mark_model_dirty(idx)

    def _save_transform_state(self):
        if not self.transform_state_path:
            print("[viewer] Transform state path not set; skipping save.")
            return

        directory = os.path.dirname(self.transform_state_path)
        if directory and not os.path.isdir(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as exc:
                print(f"[viewer] Failed to create directory for transform state: {exc}")
                return

        checkpoint_data = {}
        for idx, ckpt_path in enumerate(self.args.checkpoint):
            key = self._state_key_for_checkpoint(ckpt_path)
            trans = self.model_transforms[idx]
            scale_value = max(1e-6, float(trans["scale"]))
            trans["scale"] = scale_value
            trans["translation"] = [float(v) for v in trans["translation"]]
            self._ensure_bbox_consistency(trans)
            trans["bbox_min"] = [float(v) for v in trans["bbox_min"]]
            trans["bbox_max"] = [float(v) for v in trans["bbox_max"]]
            checkpoint_data[key] = {
                "yaw": float(trans["yaw"]),
                "pitch": float(trans["pitch"]),
                "roll": float(trans["roll"]),
                "translation": trans["translation"],
                "scale": scale_value,
                "bbox_min": trans["bbox_min"],
                "bbox_max": trans["bbox_max"],
            }

        try:
            with open(self.transform_state_path, "w", encoding="utf-8") as f:
                json.dump({"checkpoints": checkpoint_data}, f, indent=2)
            self._transform_state_cache = checkpoint_data
            print(f"[viewer] Saved transforms to {self.transform_state_path}")
        except Exception as exc:
            print(f"[viewer] Failed to save transform state: {exc}")

    def _ensure_bbox_consistency(self, trans: Dict[str, List[float]]) -> bool:
        changed = False
        for axis in range(3):
            if trans["bbox_min"][axis] > trans["bbox_max"][axis]:
                trans["bbox_min"][axis], trans["bbox_max"][axis] = (
                    trans["bbox_max"][axis],
                    trans["bbox_min"][axis],
                )
                changed = True
        return changed

    def _mark_model_dirty(self, idx: int):
        if 0 <= idx < len(self.model_render_cache):
            self.model_render_cache[idx]["dirty"] = True
            self.model_render_cache[idx]["attrs"] = None
        self.joint_render_cache["dirty"] = True
        self.joint_render_cache["attrs"] = None
        self._force_rerender = True

    def _mark_all_models_dirty(self):
        for cache in self.model_render_cache:
            cache["dirty"] = True
            cache["attrs"] = None
        self.joint_render_cache["dirty"] = True
        self.joint_render_cache["attrs"] = None
        self._force_rerender = True

    def _get_model_attrs(self, idx: int) -> Dict[str, torch.Tensor]:
        cache = self.model_render_cache[idx]
        if cache["dirty"] or cache["attrs"] is None:
            cache["attrs"] = self._compute_model_attributes(self.models[idx], self.model_transforms[idx])
            cache["dirty"] = False
        return cache["attrs"]

    def _get_joint_attrs(self) -> Dict[str, torch.Tensor]:
        cache = self.joint_render_cache
        if cache["dirty"] or cache["attrs"] is None:
            all_attrs: Dict[str, List[torch.Tensor]] = {
                "_xyz": [],
                "_normal": [],
                "_rotation": [],
                "_features_dc": [],
                "_features_rest": [],
                "_scaling": [],
                "_opacity": [],
                "_albedo": [],
                "_roughness": [],
                "_metallic": [],
            }
            for idx in range(len(self.models)):
                attrs = self._get_model_attrs(idx)
                for attr_name, tensor in attrs.items():
                    all_attrs[attr_name].append(tensor)
            joint_attrs = {attr_name: torch.cat(tensors, dim=0) for attr_name, tensors in all_attrs.items()}
            cache["attrs"] = joint_attrs
            cache["dirty"] = False
        return cache["attrs"]

    def _compute_model_attributes(self, model: GaussianModel, trans: Dict[str, object]) -> Dict[str, torch.Tensor]:
        xyz = model.get_xyz
        device = xyz.device
        dtype = xyz.dtype

        self._ensure_bbox_consistency(trans)

        yaw_rad = math.radians(trans["yaw"])
        pitch_rad = math.radians(trans["pitch"])
        roll_rad = math.radians(trans["roll"])

        R_user = torch.from_numpy(euler_to_matrix(yaw_rad, pitch_rad, roll_rad)).to(device)
        R_combined = R_user @ self.base_R_fix[:3, :3].to(device)

        translation = torch.tensor(trans["translation"], device=device, dtype=dtype)
        scale_factor = max(trans["scale"], 1e-6)
        scale_tensor_xyz = torch.tensor(scale_factor, device=device, dtype=dtype)

        rotated_xyz = (R_combined @ xyz.T).T * scale_tensor_xyz
        transformed_xyz = rotated_xyz + translation

        bbox_min = torch.tensor(trans["bbox_min"], device=device, dtype=dtype)
        bbox_max = torch.tensor(trans["bbox_max"], device=device, dtype=dtype)
        mask = ((transformed_xyz >= bbox_min) & (transformed_xyz <= bbox_max)).all(dim=1)

        normals = (R_combined @ model.get_normal.T).T
        base_rotations = build_rotation(model.get_rotation).to(device)
        combined_rotations = torch.matmul(R_combined, base_rotations)
        rotation_quat = rotation_to_quaternion(combined_rotations)

        base_scaling = model.get_scaling
        scaled_scaling = base_scaling * scale_tensor_xyz
        raw_scaling = model.scaling_inverse_activation(scaled_scaling)

        attrs = {
            "_xyz": transformed_xyz[mask],
            "_normal": normals[mask],
            "_rotation": rotation_quat[mask],
            "_features_dc": model._features_dc[mask],
            "_features_rest": model._features_rest[mask],
            "_scaling": raw_scaling[mask],
            "_opacity": model._opacity[mask],
            "_albedo": model._albedo[mask],
            "_roughness": model._roughness[mask],
            "_metallic": model._metallic[mask],
        }

        return attrs

    def _prepare_render_gaussians(self):
        if self.render_jointly:
            self._concatenate_gaussians()
        else:
            self._use_individual_gaussian()

    def _use_individual_gaussian(self):
        attrs = self._get_model_attrs(self.active_model_idx)
        for attr_name, tensor in attrs.items():
            setattr(self.render_gaussians, attr_name, tensor)

    def _concatenate_gaussians(self):
        joint_attrs = self._get_joint_attrs()
        for attr_name, tensor in joint_attrs.items():
            setattr(self.render_gaussians, attr_name, tensor)

    @torch.no_grad()
    def render_current(self) -> torch.Tensor:
        t_start = time.perf_counter()
        self._prepare_render_gaussians()
        t_after_prepare = time.perf_counter()

        view = self.camera

        rendering_result = render(
            viewpoint_camera=view,
            pc=self.render_gaussians,
            pipe=self.args.pipeline,
            bg_color=self.background,
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )
        t_after_render = time.perf_counter()

        normal_map = rendering_result["normal_map"]
        opacity_mask = rendering_result["opacity_map"]
        albedo_map = rendering_result["albedo_map"]
        roughness_map = rendering_result["roughness_map"]
        metallic_map = rendering_result["metallic_map"]

        H, W = view.image_height, view.image_width
        c2w = torch.inverse(view.world_view_transform.T)
        canonical_rays = get_canonical_rays(H, W, self.camera.FoVx, self.camera.FoVy)
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3]).sum(dim=-1).reshape(H, W, 3)
        )

        env_yaw_rad = math.radians(self.hdri_rotation_deg)
        env_cos_yaw = math.cos(env_yaw_rad)
        env_sin_yaw = math.sin(env_yaw_rad)

        x, y, z = view_dirs[..., 0], view_dirs[..., 1], view_dirs[..., 2]

        x_rotated = x * env_cos_yaw - z * env_sin_yaw
        z_rotated = x * env_sin_yaw + z * env_cos_yaw
        light_sample_dirs = torch.stack([x_rotated, y, z_rotated], dim=-1)

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
        t_after_shading = time.perf_counter()

        if self.show_env_bg:
            env_rgb = _sample_env_latlong(self.hdri, light_sample_dirs)
            if self.enable_tone:
                env_rgb = _aces_film(env_rgb)
            if self.enable_gamma:
                env_rgb = _linear_to_srgb(env_rgb)
            bg_mask = 1.0 - opacity_mask.permute(1, 2, 0).clamp(0.0, 1.0)
            render_rgb = render_rgb * (1.0 - bg_mask) + env_rgb * bg_mask
        t_after_env = time.perf_counter()

        self.profile_timings = {
            "prepare": (t_after_prepare - t_start) * 1000.0,
            "render": (t_after_render - t_after_prepare) * 1000.0,
            "shading": (t_after_shading - t_after_render) * 1000.0,
            "composite": (t_after_env - t_after_shading) * 1000.0,
            "total": (t_after_env - t_start) * 1000.0,
            "gaussian_count": float(self.render_gaussians.get_xyz.shape[0]),
        }

        return render_rgb.clamp(0.0, 1.0)

    # -------------------------
    # imgui helpers
    # -------------------------
    def _log_input(self, message: str):
        print(f"[input] {message}", flush=True)

    def _update_key_log(self, logical_name: str, is_down: bool):
        prev = self._key_state.get(logical_name, False)
        if is_down != prev:
            self._key_state[logical_name] = is_down
            key_label = self._key_names.get(logical_name, logical_name)
            state = "down" if is_down else "up"
            self._log_input(f"key {key_label} {state}")

    def _process_camera_inputs(self):
        io = imgui.get_io()
        dt = max(0.0, min(io.delta_time, self.MAX_DT))
        self._last_frame_dt = dt
        if dt <= 0.0:
            return

        camera_updated = False
        allow_keyboard = not io.want_capture_keyboard or self._render_window_focused or self._render_window_hovered
        allow_mouse = (
            not io.want_capture_mouse
            or self._render_window_hovered
            or self._render_image_hovered
            or self._render_image_active
        )

        if allow_keyboard:
            def _is_down(key_name: str) -> bool:
                key = self._key_cache.get(key_name)
                if key is None:
                    return False
                try:
                    return imgui.is_key_down(key)
                except TypeError:
                    try:
                        return imgui.is_key_down(int(key))
                    except Exception:
                        key_code = getattr(key, "value", None)
                        if key_code is None:
                            return False
                        return imgui.is_key_down(key_code)

            key_forward = _is_down("forward")
            key_back = _is_down("back")
            key_up = _is_down("up")
            key_down = _is_down("down")
            key_right = _is_down("right")
            key_left = _is_down("left")
            key_yaw_left = _is_down("yaw_left")
            key_yaw_right = _is_down("yaw_right")
            key_pitch_up = _is_down("pitch_up")
            key_pitch_down = _is_down("pitch_down")

            for logical, state in [
                ("forward", key_forward),
                ("back", key_back),
                ("up", key_up),
                ("down", key_down),
                ("right", key_right),
                ("left", key_left),
                ("yaw_left", key_yaw_left),
                ("yaw_right", key_yaw_right),
                ("pitch_up", key_pitch_up),
                ("pitch_down", key_pitch_down),
            ]:
                self._update_key_log(logical, state)

            if key_forward:
                self.camera.move_forward(+self.MOVE_SPEED * dt)
                camera_updated = True
            if key_back:
                self.camera.move_forward(-self.MOVE_SPEED * dt)
                camera_updated = True
            if key_up:
                self.camera.move_up(+self.MOVE_SPEED * dt)
                camera_updated = True
            if key_down:
                self.camera.move_up(-self.MOVE_SPEED * dt)
                camera_updated = True
            if key_right:
                self.camera.move_right(+self.MOVE_SPEED * dt)
                camera_updated = True
            if key_left:
                self.camera.move_right(-self.MOVE_SPEED * dt)
                camera_updated = True

            yaw_delta = 0.0
            pitch_delta = 0.0
            if key_yaw_left:
                yaw_delta += +self.ORBIT_KEY_SPEED * dt
            if key_yaw_right:
                yaw_delta += -self.ORBIT_KEY_SPEED * dt
            if key_pitch_up:
                pitch_delta += -self.ORBIT_KEY_SPEED * dt
            if key_pitch_down:
                pitch_delta += +self.ORBIT_KEY_SPEED * dt
            if yaw_delta or pitch_delta:
                self.camera.orbit(yaw_delta, pitch_delta)
                camera_updated = True

        def _mouse_button_code(button) -> int:
            try:
                return int(button)
            except Exception:
                if hasattr(button, "value"):
                    return int(button.value)
                return int(button)

        mouse_button = _mouse_button_code(self._mouse_left)
        if allow_mouse:
            if imgui.is_mouse_dragging(mouse_button, 0.0):
                if not self._mouse_dragging:
                    self._mouse_dragging = True
                    self._log_input("mouse drag start")
                drag_delta = imgui.get_mouse_drag_delta(mouse_button, 0.0)
                if isinstance(drag_delta, tuple):
                    dx, dy = drag_delta
                else:
                    dx = getattr(drag_delta, "x", 0.0)
                    dy = getattr(drag_delta, "y", 0.0)
                self.camera.orbit(-dx * self.MOUSE_SENSITIVITY, +dy * self.MOUSE_SENSITIVITY)
                imgui.reset_mouse_drag_delta(mouse_button)
                self._log_input(f"mouse drag delta dx={dx:.3f} dy={dy:.3f}")
                camera_updated = True
            else:
                if self._mouse_dragging:
                    self._mouse_dragging = False
                    self._log_input("mouse drag end")

        if camera_updated:
            self._force_rerender = True
            self._log_input("camera updated")

    def _ensure_camera_matches_size(self, width: int, height: int):
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.camera.image_width and height == self.camera.image_height:
            return

        aspect = float(width) / float(height)
        tan_half_y = float(self.camera.FoVy)
        tan_half_x = tan_half_y * aspect
        self.camera.FoVx = tan_half_x
        self.camera.FoVy = tan_half_y

        self.camera.projection_matrix = (
            getProjectionMatrix(
                znear=self.camera.znear,
                zfar=self.camera.zfar,
                fovX=self.camera.FoVx,
                fovY=self.camera.FoVy,
            )
            .transpose(0, 1)
            .to(self.camera.data_device)
        )
        self.camera.image = torch.zeros((3, height, width), dtype=torch.float32)
        self.camera.original_image = self.camera.image.clone()
        self.camera.image_width = width
        self.camera.image_height = height
        self.camera.update_matrices()
        self._force_rerender = True

    def _update_render_buffer(self, force: bool = False):
        if not (force or self._force_rerender):
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            img = self.render_current()
            self.render_image_rgb = np.ascontiguousarray(img.detach().cpu().numpy())
            self.render_image_rgba = tensor_to_raw_rgba(img)
            self.last_render_error = None
        except Exception as exc:
            self.last_render_error = str(exc)
        finally:
            self._force_rerender = False

    # -------------------------
    # UI drawing
    # -------------------------
    def _draw_render_window(self):
        cond_first = _imgui_cond("FirstUseEver")
        if cond_first is not None:
            imgui.set_next_window_size((self.args.width - 400, self.args.height - 40), cond_first)
            imgui.set_next_window_pos((400, 10), cond_first)

        if imgui.begin("Render"):
            if self.last_render_error:
                imgui.text_colored(f"Render error: {self.last_render_error}", 1.0, 0.2, 0.2, 1.0)
            elif self.render_image_rgb is not None:
                get_region = getattr(imgui, "get_content_region_avail", None)
                if get_region is None:
                    get_region = getattr(imgui, "get_content_region_available", None)
                avail = get_region() if get_region is not None else (self.camera.image_width, self.camera.image_height)
                available_w = max(1, int(avail.x if hasattr(avail, "x") else avail[0]))
                available_h = max(1, int(avail.y if hasattr(avail, "y") else avail[1]))
                self._ensure_camera_matches_size(available_w, available_h)
                display_size = (self.camera.image_width, self.camera.image_height)
                immvision.image_display(
                    "Relight",
                    self.render_image_rgb,
                    image_display_size=display_size,
                    refresh_image=True,
                    is_bgr_or_bgra=False,
                )
                self._render_image_hovered = bool(imgui.is_item_hovered())
                self._render_image_active = bool(imgui.is_item_active())
            else:
                imgui.text("Rendering...")
                self._render_image_hovered = False
                self._render_image_active = False
            hovered_flag_none = _imgui_hovered_flag("none", 0)
            self._render_window_hovered = bool(imgui.is_window_hovered(hovered_flag_none))
            self._render_window_focused = bool(imgui.is_window_focused())
        else:
            self._render_window_hovered = False
            self._render_window_focused = False
            self._render_image_hovered = False
            self._render_image_active = False
        imgui.end()

    def _draw_transform_controls(self):
        trans = self.model_transforms[self.active_model_idx]

        def slider(label: str, field: str, min_v: float, max_v: float):
            changed, value = imgui.slider_float(label, trans[field], min_v, max_v)
            if changed:
                trans[field] = value
                self._mark_model_dirty(self.active_model_idx)
                self._log_input(f"{label} set to {value:.3f}")

        imgui.text("Model Orientation")
        slider("Yaw (deg)", "yaw", -180.0, 180.0)
        slider("Pitch (deg)", "pitch", -180.0, 180.0)
        slider("Roll (deg)", "roll", -180.0, 180.0)

        imgui.separator()
        imgui.text("Translation")
        drag_changed, drag_values = imgui.drag_float3(
            "XYZ",
            trans["translation"],
            v_speed=0.01,
            format="%.3f",
        )
        if drag_changed:
            trans["translation"] = [float(v) for v in drag_values]
            self._mark_model_dirty(self.active_model_idx)
            self._log_input(
                "Translation set to "
                f"({trans['translation'][0]:.3f}, {trans['translation'][1]:.3f}, {trans['translation'][2]:.3f})"
            )

        imgui.separator()
        changed, scale_value = imgui.slider_float("Uniform Scale", trans["scale"], 0.0001, 10.0, format="%.3f")
        if changed:
            trans["scale"] = max(scale_value, 1e-6)
            self._mark_model_dirty(self.active_model_idx)
            self._log_input(f"Uniform Scale set to {trans['scale']:.3f}")

        imgui.separator()
        imgui.text("Bounding Box")
        bbox_min = trans["bbox_min"]
        bbox_max = trans["bbox_max"]
        drag_changed_min, drag_values_min = imgui.drag_float3(
            "BBox Min",
            bbox_min,
            v_speed=0.01,
            format="%.3f",
        )
        if drag_changed_min:
            trans["bbox_min"] = [float(v) for v in drag_values_min]
            self._ensure_bbox_consistency(trans)
            self._mark_model_dirty(self.active_model_idx)
            self._log_input(
                "BBox Min set to "
                f"({trans['bbox_min'][0]:.3f}, {trans['bbox_min'][1]:.3f}, {trans['bbox_min'][2]:.3f})"
            )
        drag_changed_max, drag_values_max = imgui.drag_float3(
            "BBox Max",
            bbox_max,
            v_speed=0.01,
            format="%.3f",
        )
        if drag_changed_max:
            trans["bbox_max"] = [float(v) for v in drag_values_max]
            self._ensure_bbox_consistency(trans)
            self._mark_model_dirty(self.active_model_idx)
            self._log_input(
                "BBox Max set to "
                f"({trans['bbox_max'][0]:.3f}, {trans['bbox_max'][1]:.3f}, {trans['bbox_max'][2]:.3f})"
            )

    def _draw_environment_controls(self):
        imgui.text("Environment")
        if imgui.begin_combo("HDRI Preset", self.hdri_label_current):
            for label in self.hdri_labels:
                is_selected = label == self.hdri_label_current
                if imgui.selectable(label, is_selected)[0]:
                    if label != self.hdri_label_current:
                        self.hdri_label_current = label
                        self._ensure_hdri(label)
                        self._mark_all_models_dirty()
                        self._log_input(f"HDRI preset set to {label}")
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        changed, rotation = imgui.slider_float("HDRI Yaw", self.hdri_rotation_deg, -180.0, 180.0, "%.1f deg")
        if changed:
            self.hdri_rotation_deg = rotation
            self._force_rerender = True
            self._log_input(f"HDRI yaw set to {rotation:.1f} degrees")

        imgui.separator()
        imgui.text("Shading")
        changed, tone_state = imgui.checkbox("ACES tone mapping", self.enable_tone)
        if changed:
            self.enable_tone = tone_state
            self._force_rerender = True
            self._log_input(f"ACES tone mapping {'enabled' if tone_state else 'disabled'}")

        changed, gamma_state = imgui.checkbox("Gamma correction (sRGB)", self.enable_gamma)
        if changed:
            self.enable_gamma = gamma_state
            self._force_rerender = True
            self._log_input(f"Gamma correction {'enabled' if gamma_state else 'disabled'}")

        changed, env_bg_state = imgui.checkbox("HDRI as background", self.show_env_bg)
        if changed:
            self.show_env_bg = env_bg_state
            self._force_rerender = True
            self._log_input(f"HDRI background {'enabled' if env_bg_state else 'disabled'}")

    def _draw_control_window(self):
        cond_first = _imgui_cond("FirstUseEver")
        if cond_first is not None:
            imgui.set_next_window_size((360, self.args.height - 40), cond_first)
            imgui.set_next_window_pos((20, 10), cond_first)

        if not imgui.begin("Controls"):
            imgui.end()
            return

        io = imgui.get_io()
        fps = 1.0 / self._last_frame_dt if self._last_frame_dt > 0 else 0.0
        imgui.text(f"FPS: {fps:.1f}")
        imgui.text(f"Frame Time: {self._last_frame_dt * 1000.0:.2f} ms")
        imgui.separator()

        render_mode_index = 1 if self.render_jointly else 0
        render_modes = ["Active Model", "Joint (Concatenated)"]
        changed, new_mode_idx = imgui.combo("Render Mode", render_mode_index, render_modes)
        if changed:
            self.render_jointly = new_mode_idx == 1
            self._mark_all_models_dirty()
            self._log_input(f"Render mode set to {render_modes[new_mode_idx]}")

        imgui.separator()
        if imgui.begin_combo("Active Model", self.model_names[self.active_model_idx]):
            for idx, name in enumerate(self.model_names):
                is_selected = idx == self.active_model_idx
                if imgui.selectable(name, is_selected)[0]:
                    if idx != self.active_model_idx:
                        self.active_model_idx = idx
                        self._mark_model_dirty(self.active_model_idx)
                        self._force_rerender = True
                        self._log_input(f"Active model set to {name}")
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.separator()
        self._draw_transform_controls()
        imgui.separator()
        self._draw_environment_controls()

        imgui.separator()
        imgui.text("Profiling")
        timings = self.profile_timings or {}
        imgui.text(f"Prepare: {timings.get('prepare', float('nan')):.2f} ms" if "prepare" in timings else "Prepare: -- ms")
        imgui.text(f"Render: {timings.get('render', float('nan')):.2f} ms" if "render" in timings else "Render: -- ms")
        imgui.text(f"Shading: {timings.get('shading', float('nan')):.2f} ms" if "shading" in timings else "Shading: -- ms")
        imgui.text(f"Composite: {timings.get('composite', float('nan')):.2f} ms" if "composite" in timings else "Composite: -- ms")
        imgui.text(f"Total: {timings.get('total', float('nan')):.2f} ms" if "total" in timings else "Total: -- ms")
        if "gaussian_count" in timings:
            imgui.text(f"Gaussians: {int(timings['gaussian_count'])}")
        else:
            imgui.text("Gaussians: --")

        imgui.separator()
        if imgui.button("Render Once"):
            self._force_rerender = True
            self._log_input("Render Once button pressed")
        imgui.same_line()
        if imgui.button("Save Transforms"):
            self._save_transform_state()
            self._log_input("Save Transforms button pressed")

        if self.transform_state_path:
            imgui.text_wrapped(f"Transforms file: {self.transform_state_path}")

        imgui.separator()
        imgui.text("Checkpoints:")
        for ckpt in self.args.checkpoint:
            imgui.text_wrapped(ckpt)

        imgui.end()

    def gui_frame(self):
        self._process_camera_inputs()
        self._update_render_buffer()
        self._draw_control_window()
        self._draw_render_window()
        self._total_frames += 1

    def run(self):
        immapp.run(
            gui_function=self.gui_frame,
            window_title="GS-IR PBR Viewer",
            window_size=[self.args.width, self.args.height],
            window_restore_previous_geometry=True,
        )


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = ArgumentParser(description="GS-IR PBR imgui_bundle Viewer")

    parser.add_argument("--config", type=str, default="config.json", help="Path to a JSON configuration file.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--checkpoint", type=str, default=None, nargs="+", help="Path(s) to GS-IR checkpoint(s) to load.")
    parser.add_argument("--hdri", type=str, default=None, help="Path to a specific latlong HDRI file (.hdr). Overrides presets.")
    parser.add_argument("--hdri_root", type=str, default=None, help="Root directory containing HDRI files for presets.")
    parser.add_argument("--env_res", type=int, default=256, help="Cubemap base resolution per face.")
    parser.add_argument("--width", type=int, default=1280, help="Viewer window width.")
    parser.add_argument("--height", type=int, default=720, help="Viewer window height.")
    parser.add_argument("--tone", action="store_true", help="Enable ACES filmic tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear->sRGB gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Use reconstructed metallic map.")
    parser.add_argument("--no_env_bg", action="store_true", help="Disable compositing HDRI as background.")
    parser.add_argument("--transform_state", type=str, default="viewer_transforms.json", help="Path to store/load per-model transforms.")

    temp_args, _ = parser.parse_known_args()

    if os.path.isfile(temp_args.config):
        print(f"[viewer] Loading arguments from: {temp_args.config}")
        with open(temp_args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        parser.set_defaults(**config_data)
    else:
        print(f"[viewer] Config file not found at '{temp_args.config}'. Using command-line arguments and defaults.")

    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("A --checkpoint must be provided either via command-line or config file.")

    if not args.hdri and not args.hdri_root:
        parser.error("Provide --hdri (specific file) or --hdri_root (directory with presets).")

    _a = type("Args", (), {})()
    _a.__dict__.update(vars(args))
    _a.pipeline = pipeline.extract(args)
    _a.sh_degree = args.sh_degree

    safe_state(getattr(args, "quiet", False))

    app = RelightViewer(_a)
    app.run()


if __name__ == "__main__":
    main()
