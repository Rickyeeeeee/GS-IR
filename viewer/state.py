from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from gaussian_renderer import GaussianModel
from pbr import CubemapLight, get_brdf_lut
from utils.general_utils import build_rotation, rotation_to_quaternion
from utils.graphics_utils import getProjectionMatrix
from utils.viewer_utils import (
    _aces_film,
    _linear_to_srgb,
    _sample_env_latlong,
    euler_to_matrix,
    get_canonical_rays,
    latlong_to_cubemap,
    read_hdr,
)
from viewer_camera import ViewerCamera


HDRI_PRESETS: List[Tuple[str, str]] = [
    ("Bridge", "bridge.hdr"),
    ("City", "city.hdr"),
    # ("Courtyard", "courtyard.hdr"),
    ("Fireplace", "fireplace.hdr"),
    ("Forest", "forest.hdr"),
    ("Interior", "interior.hdr"),
    # ("Museum", "museum.hdr"),
    ("Night", "night.hdr"),
    ("Snow", "snow.hdr"),
    ("Square", "square.hdr"),
    # ("Studio", "studio.hdr"),
    ("Sunrise", "sunrise.hdr"),
    ("Sunset", "sunset.hdr"),
    ("Tunnel", "tunnel.hdr"),
]


@dataclass
class TransformState:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: float = 1.0
    bbox_min: List[float] = field(default_factory=lambda: [-1e6, -1e6, -1e6])
    bbox_max: List[float] = field(default_factory=lambda: [1e6, 1e6, 1e6])
    bbox_yaw: float = 0.0
    bbox_pitch: float = 0.0
    bbox_roll: float = 0.0


@dataclass
class PointLightState:
    enabled: bool = False
    position: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    intensity: List[float] = field(default_factory=lambda: [100.0, 100.0, 100.0])
    enable_shadow: bool = False
    shadow_bias: float = 0.3
    shadow_resolution: int = 2048


class ViewerState:
    """Keeps every bit of long-lived state for the viewer."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models: List[GaussianModel] = []
        self.model_names: List[str] = []
        self._load_models()

        self.render_gaussians = GaussianModel(args.sh_degree)
        self.model_transforms: List[Dict[str, float | List[float]]] = [
            TransformState().__dict__.copy() for _ in self.models
        ]
        self.active_model_idx = 0
        self.render_jointly = False

        self.transform_state_path = os.path.abspath(getattr(args, "transform_state", "viewer_transforms.json"))
        self._transform_state_cache: Dict[str, Dict[str, object]] = {}
        self.model_render_cache: List[Dict[str, object]] = [{"attrs": None, "dirty": True} for _ in self.models]
        self.joint_render_cache: Dict[str, object] = {"attrs": None, "dirty": True}
        self.profile_timings: Dict[str, float] = {}

        self.base_R_fix = torch.from_numpy(
            euler_to_matrix(
                torch.deg2rad(torch.tensor(0.0)),
                torch.deg2rad(torch.tensor(90.0)),
                torch.deg2rad(torch.tensor(0.0)),
            )
        ).cuda()

        self.hdri_presets: List[Tuple[str, str]] = []
        if args.hdri_root and os.path.isdir(args.hdri_root):
            for label, filename in HDRI_PRESETS:
                self.hdri_presets.append((label, os.path.join(args.hdri_root, filename)))

        if args.hdri and not any(os.path.normpath(p) == os.path.normpath(args.hdri) for _, p in self.hdri_presets):
            self.hdri_presets.insert(0, ("(from --hdri)", args.hdri))

        if not self.hdri_presets:
            raise ValueError("No HDRIs available. Provide --hdri or --hdri_root.")

        self.hdri_labels = [label for label, _ in self.hdri_presets]
        self.hdri_paths = {label: path for label, path in self.hdri_presets}
        self.hdri_label_current = self._label_from_path(args.hdri) if args.hdri else self.hdri_labels[0]
        self.hdri_cache_latlong: Dict[str, torch.Tensor] = {}
        self.hdri_cache_cubemap: Dict[str, torch.Tensor] = {}
        self.hdri_rotation_deg = 0.0

        self.light = CubemapLight(base_res=args.env_res).to(self.device)
        self.brdf_lut = get_brdf_lut().to(self.device)
        self.ensure_hdri(self.hdri_label_current)
        self.light.eval()

        self.camera = ViewerCamera(
            FoVx=1.6609910500147054,
            FoVy=1.103431263015383,
            W=args.width - 320,
            H=args.height - 40,
            data_device="cuda",
        )
        self.target = self.models[0].get_xyz.mean(dim=0).detach().cpu().numpy()
        self.camera.look_at(self.target, distance=1.0)

        self.point_light = PointLightState(
            position=list(self.camera.camera_center.detach().cpu().tolist())
        )
        self._point_shadow_cache: Dict[str, object] = {
            "depth_cubemap": None,
            "opacity_cubemap": None,
            "position": None,
            "resolution": None,
            "dirty": True,
        }

        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        self.enable_tone = args.tone
        self.enable_gamma = args.gamma
        self.enable_metallic = args.metallic
        self.show_env_bg = not getattr(args, "no_env_bg", False)

        self.hdri = self.hdri_cache_latlong[self.hdri_label_current]

        self.load_transform_state()

    # ---- model / transform helpers -------------------------------------------------
    def _load_models(self) -> None:
        for idx, ckpt_path in enumerate(self.args.checkpoint):
            model_name = f"Model {idx + 1} ({os.path.basename(ckpt_path)})"
            print(f"[viewer] Loading: {model_name}")
            model = GaussianModel(self.args.sh_degree)
            checkpoint = torch.load(ckpt_path)
            model_params = checkpoint.get("gaussians", checkpoint) if isinstance(checkpoint, dict) else checkpoint[0]
            model.restore(model_params)
            self.models.append(model)
            self.model_names.append(model_name)

    def _label_from_path(self, path: str | None) -> str:
        if not path:
            return self.hdri_labels[0]
        norm = os.path.normpath(path)
        for label, preset_path in self.hdri_presets:
            if os.path.normpath(preset_path) == norm:
                return label
        return self.hdri_labels[0]

    def ensure_hdri(self, label: str) -> None:
        path = self.hdri_paths[label]
        if label not in self.hdri_cache_latlong:
            hdri_np = read_hdr(path)
            self.hdri_cache_latlong[label] = torch.from_numpy(hdri_np).to(self.device)
        latlong = self.hdri_cache_latlong[label]

        if label not in self.hdri_cache_cubemap:
            self.hdri_cache_cubemap[label] = latlong_to_cubemap(latlong, [self.args.env_res, self.args.env_res])

        self.hdri = latlong
        self.light.base.data = self.hdri_cache_cubemap[label]
        self.light.build_mips()

    def _state_key_for_checkpoint(self, ckpt_path: str) -> str:
        abs_path = os.path.abspath(ckpt_path)
        scene_dir = os.path.basename(os.path.dirname(abs_path))
        if not scene_dir:
            scene_dir = os.path.splitext(os.path.basename(abs_path))[0]
        return scene_dir

    def load_transform_state(self) -> None:
        if not os.path.isfile(self.transform_state_path):
            return
        with open(self.transform_state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        checkpoint_data = payload.get("checkpoints", {})
        self._transform_state_cache = checkpoint_data

        for idx, ckpt_path in enumerate(self.args.checkpoint):
            key = self._state_key_for_checkpoint(ckpt_path)
            saved = checkpoint_data.get(key) or checkpoint_data.get(os.path.abspath(ckpt_path))
            if not isinstance(saved, dict):
                continue
            trans = self.model_transforms[idx]
            trans["yaw"] = float(saved.get("yaw", trans["yaw"]))
            trans["pitch"] = float(saved.get("pitch", trans["pitch"]))
            trans["roll"] = float(saved.get("roll", trans["roll"]))
            translation = saved.get("translation", trans["translation"])
            if isinstance(translation, (list, tuple)) and len(translation) == 3:
                trans["translation"] = [float(v) for v in translation]
            trans["scale"] = max(1e-6, float(saved.get("scale", trans["scale"])))
            bbox_min = saved.get("bbox_min")
            if isinstance(bbox_min, (list, tuple)) and len(bbox_min) == 3:
                trans["bbox_min"] = [float(v) for v in bbox_min]
            bbox_max = saved.get("bbox_max")
            if isinstance(bbox_max, (list, tuple)) and len(bbox_max) == 3:
                trans["bbox_max"] = [float(v) for v in bbox_max]
            trans["bbox_yaw"] = float(saved.get("bbox_yaw", trans["bbox_yaw"]))
            trans["bbox_pitch"] = float(saved.get("bbox_pitch", trans["bbox_pitch"]))
            trans["bbox_roll"] = float(saved.get("bbox_roll", trans["bbox_roll"]))

    def save_transform_state(self) -> None:
        checkpoint_data = {}
        for ckpt_path, trans in zip(self.args.checkpoint, self.model_transforms):
            key = self._state_key_for_checkpoint(ckpt_path)
            checkpoint_data[key] = {
                "yaw": float(trans["yaw"]),
                "pitch": float(trans["pitch"]),
                "roll": float(trans["roll"]),
                "translation": [float(v) for v in trans["translation"]],
                "scale": float(trans["scale"]),
                "bbox_min": [float(v) for v in trans["bbox_min"]],
                "bbox_max": [float(v) for v in trans["bbox_max"]],
                "bbox_yaw": float(trans.get("bbox_yaw", 0.0)),
                "bbox_pitch": float(trans.get("bbox_pitch", 0.0)),
                "bbox_roll": float(trans.get("bbox_roll", 0.0)),
            }

        with open(self.transform_state_path, "w", encoding="utf-8") as handle:
            json.dump({"checkpoints": checkpoint_data}, handle, indent=2)
        self._transform_state_cache = checkpoint_data
        print(f"[viewer] Saved transforms to {self.transform_state_path}")

    def ensure_bbox_consistency(self, trans: Dict[str, List[float]]) -> None:
        for axis in range(3):
            if trans["bbox_min"][axis] > trans["bbox_max"][axis]:
                trans["bbox_min"][axis], trans["bbox_max"][axis] = (
                    trans["bbox_max"][axis],
                    trans["bbox_min"][axis],
                )

    def mark_model_dirty(self, idx: int) -> None:
        if 0 <= idx < len(self.model_render_cache):
            self.model_render_cache[idx]["dirty"] = True
            self.model_render_cache[idx]["attrs"] = None
        self.joint_render_cache["dirty"] = True
        self.joint_render_cache["attrs"] = None
        self.mark_point_light_dirty()

    def mark_all_models_dirty(self) -> None:
        for cache in self.model_render_cache:
            cache["dirty"] = True
            cache["attrs"] = None
        self.joint_render_cache["dirty"] = True
        self.joint_render_cache["attrs"] = None
        self.mark_point_light_dirty()

    def mark_point_light_dirty(self) -> None:
        self._point_shadow_cache["dirty"] = True

    def get_point_light_position_tensor(self) -> torch.Tensor:
        return torch.tensor(self.point_light.position, device=self.device, dtype=torch.float32)

    def get_point_light_intensity_tensor(self) -> torch.Tensor:
        return torch.tensor(self.point_light.intensity, device=self.device, dtype=torch.float32)

    def get_point_light_shadow_cache(self) -> Dict[str, object]:
        return self._point_shadow_cache

    # ---- rendering helpers ---------------------------------------------------------
    def _compute_model_attributes(self, model: GaussianModel, trans: Dict[str, object]) -> Dict[str, torch.Tensor]:
        xyz = model.get_xyz
        device = xyz.device
        dtype = xyz.dtype

        self.ensure_bbox_consistency(trans)

        yaw_rad = math.radians(trans["yaw"])
        pitch_rad = math.radians(trans["pitch"])
        roll_rad = math.radians(trans["roll"])

        R_user = torch.from_numpy(euler_to_matrix(yaw_rad, pitch_rad, roll_rad)).to(device)
        R_combined = R_user @ self.base_R_fix[:3, :3].to(device)

        translation = torch.tensor(trans["translation"], device=device, dtype=dtype)
        scale_factor = max(trans["scale"], 1e-6)
        scale_tensor = torch.tensor(scale_factor, device=device, dtype=dtype)

        rotated_xyz = (R_combined @ xyz.T).T * scale_tensor
        transformed_xyz = rotated_xyz + translation

        bbox_min = torch.tensor(trans["bbox_min"], device=device, dtype=dtype)
        bbox_max = torch.tensor(trans["bbox_max"], device=device, dtype=dtype)
        bbox_yaw = math.radians(float(trans.get("bbox_yaw", 0.0)))
        bbox_pitch = math.radians(float(trans.get("bbox_pitch", 0.0)))
        bbox_roll = math.radians(float(trans.get("bbox_roll", 0.0)))
        bbox_rotation = torch.from_numpy(
            euler_to_matrix(bbox_yaw, bbox_pitch, bbox_roll)
        ).to(device=device, dtype=dtype)
        local_points = torch.matmul(xyz, bbox_rotation.t())
        mask = ((local_points >= bbox_min) & (local_points <= bbox_max)).all(dim=1)

        normals = (R_combined @ model.get_normal.T).T
        base_rotations = build_rotation(model.get_rotation).to(device)
        combined_rotations = torch.matmul(R_combined, base_rotations)
        rotation_quat = rotation_to_quaternion(combined_rotations)

        scaled_scaling = model.get_scaling * scale_tensor
        raw_scaling = model.scaling_inverse_activation(scaled_scaling)

        return {
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

    def _get_model_attrs(self, idx: int) -> Dict[str, torch.Tensor]:
        cache = self.model_render_cache[idx]
        if cache["dirty"] or cache["attrs"] is None:
            cache["attrs"] = self._compute_model_attributes(self.models[idx], self.model_transforms[idx])
            cache["dirty"] = False
        return cache["attrs"]

    def _get_joint_attrs(self) -> Dict[str, torch.Tensor]:
        cache = self.joint_render_cache
        if cache["dirty"] or cache["attrs"] is None:
            buckets: Dict[str, List[torch.Tensor]] = {
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
                for name, tensor in attrs.items():
                    buckets[name].append(tensor)
            cache["attrs"] = {name: torch.cat(tensors, dim=0) for name, tensors in buckets.items()}
            cache["dirty"] = False
        return cache["attrs"]

    def prepare_render_gaussians(self) -> GaussianModel:
        if self.render_jointly:
            attrs = self._get_joint_attrs()
        else:
            attrs = self._get_model_attrs(self.active_model_idx)
        for attr_name, tensor in attrs.items():
            setattr(self.render_gaussians, attr_name, tensor)
        return self.render_gaussians

    # ---- shading helpers -----------------------------------------------------------
    def composite_render(
        self,
        render_rgb: torch.Tensor,
        opacity_mask: torch.Tensor,
        light_sample_dirs: torch.Tensor,
        enable_tone: bool,
        enable_gamma: bool,
    ) -> torch.Tensor:
        if not self.show_env_bg:
            return render_rgb
        env_rgb = _sample_env_latlong(self.hdri, light_sample_dirs)
        if enable_tone:
            env_rgb = _aces_film(env_rgb)
        if enable_gamma:
            env_rgb = _linear_to_srgb(env_rgb)
        bg_mask = 1.0 - opacity_mask.permute(1, 2, 0).clamp(0.0, 1.0)
        return render_rgb * (1.0 - bg_mask) + env_rgb * bg_mask

    def update_camera_resolution(self, width: int, height: int) -> None:
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
