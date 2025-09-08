# gsir_dearpygui_viewer.py
#
# Dear PyGui viewer (PBR only):
# 1) Loads a GSIR Gaussian checkpoint
# 2) Manual look-at camera (MiniCam schema matches your renderer)
# 3) PBR relighting with an HDRI cubemap (no 3DGS shaded fallback)
# 4) Mouse orbit:
#    - LMB drag: orbit (yaw/pitch)
#    - RMB drag: pan
#    - Wheel:    zoom
#
# Example:
# python gsir_dearpygui_viewer.py \
#   --checkpoint output/garden-linear/chkpnt35000.pth \
#   --hdri assets/studio_small_08_4k.hdr \
#   --width 800 --height 600 --fov_deg 60 \
#   --eye 0 0 3 --center 0 0 0 --up 0 1 0 \
#   --tone --gamma --metallic

import os
import sys
import math
from argparse import ArgumentParser
from typing import Dict, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
import dearpygui.dearpygui as dpg
import nvdiffrast.torch as dr

# --- project imports ---
from arguments import PipelineParams
from gaussian_renderer import GaussianModel, render
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from pbr import CubemapLight, get_brdf_lut, pbr_shading

try:
    import cv2
except Exception:
    cv2 = None


# ---------------- Camera helpers ----------------

def normalize_t(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return v / (torch.linalg.norm(v) + eps)

def lookat_to_RT(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    z = normalize_t(eye - center)          # forward
    x = normalize_t(torch.cross(up, z))    # right
    y = torch.cross(z, x)                  # up
    R = torch.stack([x, y, z], dim=0)
    T = -R @ eye
    return R.detach().cpu().numpy().astype(np.float32), T.detach().cpu().numpy().astype(np.float32)

def fovx_from_fovy(fovy_rad: float, aspect: float) -> float:
    return 2.0 * math.atan(aspect * math.tan(0.5 * fovy_rad))

class MiniCam:
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


# ---------------- Image helpers ----------------

def tensor_to_rgba_list(img: Union[torch.Tensor, np.ndarray]) -> Tuple[int, int, list]:
    if isinstance(img, torch.Tensor):
        img = img.detach().clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
    else:
        img = np.clip(img, 0.0, 1.0)
    h, w, _ = img.shape
    alpha = np.ones((h, w, 1), dtype=img.dtype)
    rgba = np.concatenate([img, alpha], axis=-1)
    return w, h, rgba.reshape(-1).tolist()

def make_bg_for_renderer(bg_color_val: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([bg_color_val, bg_color_val, bg_color_val], dtype=torch.float32, device=device)


# ---------------- PBR helpers (copied in) ----------------

def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if   s == 0: rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=latlong_map.device)
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=latlong_map.device),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=latlong_map.device),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)
        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)
        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode="linear")[0]
    return cubemap

def get_canonical_rays(H: int, W: int, tan_fovx: float, tan_fovy: float, device: torch.device) -> torch.Tensor:
    """
    Returns camera-space directions (not normalized) with z-forward convention.
    `tan_fov*` should be tan(FOV*/2).
    """
    cen_x = W / 2.0
    cen_y = H / 2.0
    fx = W / (2.0 * tan_fovx)
    fy = H / (2.0 * tan_fovy)
    x, y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")
    x = x.flatten()
    y = y.flatten()
    camera_dirs = F.pad(
        torch.stack([(x - cen_x + 0.5) / fx, (y - cen_y + 0.5) / fy], dim=-1),
        (0, 1),
        value=1.0,
    )  # [H*W,3]
    return camera_dirs


# ---------------- Loader & PBR render ----------------

@torch.no_grad()
def load_gaussians_from_ckpt(checkpoint_path: str, sh_degree: int, device: torch.device) -> GaussianModel:
    gaussians = GaussianModel(sh_degree)
    ckpt = torch.load(checkpoint_path, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(ckpt, tuple):
        model_params = ckpt[0]
    elif isinstance(ckpt, dict):
        model_params = ckpt.get("gaussians", ckpt.get("state_dict", ckpt))
    else:
        raise TypeError("Unsupported checkpoint format for GSIR checkpoint.")
    gaussians.restore(model_params)
    return gaussians

@torch.no_grad()
def render_view_pbr(
    cam: MiniCam,
    gaussians: GaussianModel,
    pipeline,
    bg_color_val: float,
    hdri_path: str,
    tone: bool,
    gamma: bool,
    metallic: bool,
) -> torch.Tensor:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required to read HDR/EXR env maps. Please install opencv-python-headless.")
    if not (isinstance(hdri_path, str) and os.path.isfile(hdri_path)):
        raise FileNotFoundError(f"HDRI file not found: {hdri_path}")

    device = cam.world_view_transform.device
    H, W = cam.image_height, cam.image_width

    # 1) GS rasterizer with normals/albedo/roughness/metallic
    bg_vec = make_bg_for_renderer(bg_color_val, device)
    result: Dict[str, torch.Tensor] = render(
        viewpoint_camera=cam,
        pc=gaussians,
        pipe=pipeline,
        bg_color=bg_vec,
        inference=True,
        pad_normal=True,
        derive_normal=True,
    )

    normal_map    = result["normal_map"]      # [3,H,W]
    normal_mask   = result["normal_mask"]     # [1,H,W]
    albedo_map    = result["albedo_map"]      # [3,H,W]
    roughness_map = result["roughness_map"]   # [1,H,W]
    metallic_map  = result["metallic_map"]    # [1,H,W]

    # 2) Build env cubemap from latlong HDR/EXR
    bgr = cv2.imread(hdri_path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"Failed to read HDRI: {hdri_path}")
    hdri = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hdri = torch.from_numpy(hdri).to(device=device, dtype=torch.float32)

    res = 256
    cubemap = CubemapLight(base_res=res).to(device)
    cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
    cubemap.eval()
    cubemap.build_mips()
    brdf_lut = get_brdf_lut().to(device)

    # 3) Per-pixel view directions (world space)
    tan_fovx = math.tan(cam.FoVx * 0.5)
    tan_fovy = math.tan(cam.FoVy * 0.5)
    rays_cam = get_canonical_rays(H=H, W=W, tan_fovx=tan_fovx, tan_fovy=tan_fovy, device=device)  # [H*W,3]
    rays_cam = F.normalize(rays_cam, p=2, dim=-1)
    c2w = torch.inverse(cam.world_view_transform.T)  # [4,4]
    view_dirs = -((rays_cam[:, None, :] * c2w[None, :3, :3]).sum(dim=-1)).reshape(H, W, 3)  # [H,W,3]

    # 4) PBR shading
    pbr = pbr_shading(
        light=cubemap,
        normals=normal_map.permute(1, 2, 0),                  # [H,W,3]
        view_dirs=view_dirs,                                   # [H,W,3]
        mask=normal_mask.permute(1, 2, 0),                     # [H,W,1]
        albedo=albedo_map.permute(1, 2, 0),                    # [H,W,3]
        roughness=roughness_map.permute(1, 2, 0),              # [H,W,1]
        metallic=metallic_map.permute(1, 2, 0) if metallic else None,  # [H,W,1] or None
        tone=tone,
        gamma=gamma,
        brdf_lut=brdf_lut,
    )
    rgb = pbr["render_rgb"].clamp(0.0, 1.0).permute(2, 0, 1)  # [3,H,W]
    return rgb


# ---------------- App ----------------

class GSIRViewerApp:
    def __init__(self, cam: MiniCam, gaussians: GaussianModel, pipeline,
                 bg: float, hdri_path: str, tone: bool, gamma: bool, metallic: bool):
        self.cam = cam
        self.gaussians = gaussians
        self.pipeline = pipeline# gsir_pbr_viewer.py
#
# Self-contained PBR + Viewer (no relight.py import, no fallbacks).
# - PBR: Cook–Torrance GGX with directional light (no depth/points required).
# - Uses GS rasterizer to fetch: albedo, normal, roughness, metallic.
# - Orbit camera: LMB orbit, RMB pan, wheel zoom.
#
# Requirements in your env:
#   - dearpygui
#   - torch, numpy
#   - your project modules: arguments, gaussian_renderer, utils.graphics_utils

import os
import sys
import math
from argparse import ArgumentParser
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import dearpygui.dearpygui as dpg

# ---- project imports (unchanged) ----
from arguments import PipelineParams
from gaussian_renderer import GaussianModel, render
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


# ===========================
# Camera + tiny math helpers
# ===========================

def _normalize_t(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return v / (torch.linalg.norm(v) + eps)

def _lookat_to_RT(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    z = _normalize_t(eye - center)               # forward
    x = _normalize_t(torch.cross(up, z))         # right
    y = torch.cross(z, x)                        # up
    R = torch.stack([x, y, z], dim=0)
    T = -R @ eye
    return (
        R.detach().cpu().numpy().astype(np.float32),
        T.detach().cpu().numpy().astype(np.float32),
    )

def _fovx_from_fovy(fovy_rad: float, aspect: float) -> float:
    return 2.0 * math.atan(aspect * math.tan(0.5 * fovy_rad))

class MiniCam:
    """Minimal camera matching your renderer expectations."""
    def __init__(
        self,
        width: int, height: int,
        fovy: float, fovx: float,
        znear: float, zfar: float,
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
    device: torch.device, znear: float = 0.01, zfar: float = 100.0
) -> MiniCam:
    R_np, T_np = _lookat_to_RT(eye, center, up)
    trans = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = 1.0
    fovy = math.radians(fov_deg)
    aspect = float(width) / float(height)
    fovx = _fovx_from_fovy(fovy, aspect)
    w2v = torch.tensor(getWorld2View2(R_np, T_np, trans, scale), dtype=torch.float32, device=device).transpose(0, 1)
    proj = torch.tensor(getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy),
                        dtype=torch.float32, device=device).transpose(0, 1)
    return MiniCam(width, height, fovy, fovx, znear, zfar, w2v, proj, device)


# ===========================
# PBR (Cook–Torrance, GGX)
# ===========================

def _saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)

def _DistributionGGX(n: torch.Tensor, h: torch.Tensor, rough: torch.Tensor) -> torch.Tensor:
    # n,h: [...,3], rough: [...,1]
    a  = rough * rough
    a2 = a * a
    NoH  = _saturate_dot(n, h)
    NoH2 = NoH * NoH
    nom = a2
    denom = (NoH2 * (a2 - 1.0) + 1.0)
    denom = math.pi * denom * denom
    return nom / (denom + 1e-7)

def _GeometrySchlickGGX(NoV: torch.Tensor, rough: torch.Tensor) -> torch.Tensor:
    r = rough + 1.0
    k = (r * r) / 8.0
    return NoV / (NoV * (1.0 - k) + k + 1e-7)

def _GeometrySmith(n: torch.Tensor, v: torch.Tensor, l: torch.Tensor, rough: torch.Tensor) -> torch.Tensor:
    NoV = _saturate_dot(n, v)
    NoL = _saturate_dot(n, l)
    return _GeometrySchlickGGX(NoV, rough) * _GeometrySchlickGGX(NoL, rough)

def _fresnel_schlick(HoV: torch.Tensor, F0: torch.Tensor) -> torch.Tensor:
    return F0 + (1.0 - F0) * torch.pow((1.0 - HoV).clamp(0.0, 1.0), 5.0)

def _apply_tone_gamma(img: torch.Tensor, tone: bool, gamma: bool) -> torch.Tensor:
    # img: [H,W,3] linear
    out = img
    if tone:
        # simple ACES-ish curve
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        out = (out * (a * out + b)) / (out * (c * out + d) + e + 1e-8)
    if gamma:
        out = out.clamp(0, 1) ** (1.0 / 2.2)
    return out


def pbr_directional(
    normals: torch.Tensor,     # [H,W,3], unit
    view_dirs: torch.Tensor,   # [H,W,3], unit (from surface to camera)
    albedo: torch.Tensor,      # [H,W,3] in [0,1]
    roughness: torch.Tensor,   # [H,W,1] in [0,1]
    mask: torch.Tensor,        # [H,W,1] bool-ish
    *, light_dir_world: torch.Tensor,   # [3], unit
    light_intensity: float,             # scalar radiance multiplier
    metallic_map: Union[torch.Tensor, None] = None,  # [H,W,1]
    tone: bool = False,
    gamma: bool = False,
) -> torch.Tensor:
    """
    Simple Cook–Torrance (GGX) directional-light PBR.
    Returns [H,W,3] in [0,1] (after tone/gamma if toggled).
    """
    H, W, _ = normals.shape
    device = normals.device

    l = light_dir_world.reshape(1, 1, 3).expand(H, W, 3)     # [H,W,3], already normalized
    v = view_dirs                                            # [H,W,3]
    n = F.normalize(normals, p=2, dim=-1)                    # [H,W,3]
    h = F.normalize(l + v, p=2, dim=-1)                      # [H,W,3]

    NoV = _saturate_dot(n, v)  # [H,W,1]
    NoL = _saturate_dot(n, l)  # [H,W,1]
    HoV = _saturate_dot(h, v)  # [H,W,1]

    F0 = torch.ones_like(albedo) * 0.04                      # dielectric base
    if metallic_map is not None:
        F0 = (1.0 - metallic_map) * 0.04 + albedo * metallic_map  # artist-friendly

    D  = _DistributionGGX(n, h, roughness)                   # [H,W,1] broadcast to 3
    G  = _GeometrySmith(n, v, l, roughness)                  # [H,W,1]
    fresnel  = _fresnel_schlick(HoV, F0)                           # [H,W,3]

    spec = (D * G).expand_as(fresnel) * fresnel / (4.0 * (NoV * NoL + 1e-7))  # [H,W,3]

    kd = (1.0 - fresnel)                                           # [H,W,3]
    if metallic_map is not None:
        kd = kd * (1.0 - metallic_map)

    radiance = light_intensity                               # scalar radiance
    color = (kd * albedo / math.pi + spec) * radiance * NoL  # [H,W,3]
    color = torch.where(mask > 0.5, color, torch.zeros_like(color))

    color = _apply_tone_gamma(color, tone=tone, gamma=gamma).clamp(0.0, 1.0)
    return color


# ===========================
# Render path (PBR only)
# ===========================

@torch.no_grad()
def render_pbr_view(
    cam: MiniCam,
    gaussians: GaussianModel,
    pipeline,
    *,
    bg: float,
    light_dir: Tuple[float, float, float],
    light_intensity: float,
    tone: bool,
    gamma: bool,
    use_metallic: bool,
) -> torch.Tensor:
    """
    PBR render entry point (directional light). Returns [3,H,W].
    """
    device = cam.world_view_transform.device
    H, W = cam.image_height, cam.image_width

    bg_vec = torch.tensor([bg, bg, bg], dtype=torch.float32, device=device)

    # Ask rasterizer for material buffers
    out: Dict[str, torch.Tensor] = render(
        viewpoint_camera=cam,
        pc=gaussians,
        pipe=pipeline,
        bg_color=bg_vec,
        inference=True,
        pad_normal=True,
        derive_normal=True,
    )
    # expected: 'render' (old shaded), 'normal_map','normal_mask','albedo_map','roughness_map','metallic_map'
    nmap   = out["normal_map"].permute(1, 2, 0)      # [H,W,3]
    nmask  = out["normal_mask"].permute(1, 2, 0)     # [H,W,1]
    albedo = out["albedo_map"].permute(1, 2, 0)      # [H,W,3]
    rough  = out["roughness_map"].permute(1, 2, 0)   # [H,W,1]
    metal  = out["metallic_map"].permute(1, 2, 0) if use_metallic else None  # [H,W,1] or None

    # View dirs (world): from surface toward camera.
    # With only normals available, we approximate by using the camera forward per pixel.
    # A good approximation for viewdirs is using -Z_cam rotated to world; for perspective,
    # direction varies slightly across the image, but this simple version is robust.
    cam_to_world = torch.inverse(cam.world_view_transform.T)[:3, :3]       # [3,3]
    v_world = (-cam_to_world[:, 2]).reshape(1, 1, 3).expand(H, W, 3)       # [H,W,3]
    v_world = F.normalize(v_world, p=2, dim=-1)

    light_dir_world = torch.tensor(light_dir, dtype=torch.float32, device=device)
    light_dir_world = F.normalize(light_dir_world, p=2, dim=-1)

    rgb = pbr_directional(
        normals=nmap, view_dirs=v_world,
        albedo=albedo, roughness=rough, mask=nmask,
        light_dir_world=light_dir_world,
        light_intensity=float(light_intensity),
        metallic_map=metal,
        tone=bool(tone), gamma=bool(gamma),
    )  # [H,W,3]
    return rgb.permute(2, 0, 1)  # [3,H,W]


# ===========================
# Viewer (Dear PyGui)
# ===========================

def _tensor_to_rgba_list(img: Union[torch.Tensor, np.ndarray]) -> Tuple[int, int, list]:
    if isinstance(img, torch.Tensor):
        img = img.detach().clamp(0,1).permute(1,2,0).contiguous().cpu().numpy()
    else:
        img = np.clip(img, 0.0, 1.0)
    h, w, _ = img.shape
    a = np.ones((h, w, 1), dtype=img.dtype)
    rgba = np.concatenate([img, a], axis=-1)
    return w, h, rgba.reshape(-1).tolist()

class App:
    def __init__(
        self,
        cam: MiniCam,
        gaussians: GaussianModel,
        pipeline,
        *,
        bg: float,
        light_dir: Tuple[float, float, float],
        light_intensity: float,
        tone: bool,
        gamma: bool,
        use_metallic: bool,
    ):
        self.cam = cam
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.bg = float(bg)
        self.light_dir = tuple(light_dir)
        self.light_intensity = float(light_intensity)
        self.tone = bool(tone)
        self.gamma = bool(gamma)
        self.use_metallic = bool(use_metallic)

        self.device = self.cam.world_view_transform.device
        self.texture_id = None
        self.tex_width = 0
        self.tex_height = 0

        # Orbit rig state from initial camera center/target
        self.target = torch.mean(self.gaussians.get_xyz, dim=0)
        v = (self.cam.camera_center - self.target).detach().cpu().numpy()
        self.radius = float(np.linalg.norm(v) + 1e-9)
        self.yaw   = math.atan2(v[0], v[2])
        self.pitch = math.atan2(v[1], math.sqrt(v[0]**2 + v[2]**2))

        self._lmb_down = False
        self._rmb_down = False
        self.rotate_sensitivity = 0.0005
        self.pan_sensitivity = 0.00015
        self.zoom_sensitivity = 0.01

    # Camera rebuild
    def _eye_from_orbit(self) -> torch.Tensor:
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        dir_world = torch.tensor([sy * cp, sp, cy * cp], dtype=torch.float32, device=self.device)
        return self.target + self.radius * dir_world

    def _rebuild_camera(self):
        eye = self._eye_from_orbit()
        self.cam = build_minicam(
            width=self.cam.image_width, height=self.cam.image_height,
            fov_deg=math.degrees(self.cam.FoVy),
            eye=eye, center=self.target,
            up=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
            device=self.device, znear=self.cam.znear, zfar=self.cam.zfar,
        )

    # Render & upload
    def _render_and_upload(self):
        img = render_pbr_view(
            self.cam, self.gaussians, self.pipeline,
            bg=self.bg,
            light_dir=self.light_dir,
            light_intensity=self.light_intensity,
            tone=self.tone, gamma=self.gamma,
            use_metallic=self.use_metallic,
        )
        _, _, rgba = _tensor_to_rgba_list(img)
        dpg.set_value(self.texture_id, rgba)

    # Mouse handlers
    def _on_mouse_down(self, button, x, y):
        if button == 0: self._lmb_down = True
        elif button == 1: self._rmb_down = True

    def _on_mouse_up(self, button, x, y):
        if button == 0: self._lmb_down = False
        elif button == 1: self._rmb_down = False

    def _on_drag(self, button, data):
        dx, dy = float(data[1]), float(data[2])
        if button == 0 and self._lmb_down:
            self.yaw   -= dx * self.rotate_sensitivity
            self.pitch -= dy * self.rotate_sensitivity
            self.pitch = max(-math.radians(89.0), min(math.radians(89.0), self.pitch))
            self._rebuild_camera()
            self._render_and_upload()
        elif button == 1 and self._rmb_down:
            w2v = self.cam.world_view_transform
            right = w2v[0, :3]; up = w2v[1, :3]
            pan = self.radius * self.pan_sensitivity
            self.target = self.target - right * (dx * pan) + up * (dy * pan)
            self._rebuild_camera()
            self._render_and_upload()

    def _on_wheel(self, delta):
        self.radius = max(1e-3, self.radius * math.exp(-self.zoom_sensitivity * float(delta)))
        self._rebuild_camera()
        self._render_and_upload()

    # DPG setup/run
    def run(self):
        dpg.create_context()

        img = render_pbr_view(
            self.cam, self.gaussians, self.pipeline,
            bg=self.bg, light_dir=self.light_dir,
            light_intensity=self.light_intensity,
            tone=self.tone, gamma=self.gamma,
            use_metallic=self.use_metallic,
        )
        w, h, rgba = _tensor_to_rgba_list(img)
        self.tex_width, self.tex_height = w, h

        dpg.create_viewport(
            title="GSIR PBR Viewer (Directional Light)",
            width=max(800, self.tex_width + 200),
            height=max(600, self.tex_height + 200),
        )
        with dpg.texture_registry(show=False):
            self.texture_id = dpg.add_dynamic_texture(self.tex_width, self.tex_height, rgba)
        with dpg.window(label="Viewport", width=-1, height=-1, tag="Viewport"):
            with dpg.child_window(width=-1, height=-50, border=False):
                dpg.add_image(self.texture_id)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Re-render", callback=lambda: self._render_and_upload())
                dpg.add_text(f"tone={'on' if self.tone else 'off'} | gamma={'on' if self.gamma else 'off'} | metallic={'on' if self.use_metallic else 'off'}")
                dpg.add_text(f"light_dir={tuple(round(x,3) for x in self.light_dir)}  intensity={self.light_intensity}")

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=lambda s, a, u: self._on_mouse_down(a, *dpg.get_mouse_pos()))
            dpg.add_mouse_release_handler(callback=lambda s, a, u: self._on_mouse_up(a, *dpg.get_mouse_pos()))
            dpg.add_mouse_drag_handler(button=0, callback=lambda s, a, u: self._on_drag(0, a))
            dpg.add_mouse_drag_handler(button=1, callback=lambda s, a, u: self._on_drag(1, a))
            dpg.add_mouse_wheel_handler(callback=lambda s, a, u: self._on_wheel(a))

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Viewport", True)
        dpg.start_dearpygui()
        dpg.destroy_context()


# ===========================
# Main
# ===========================

def main():
    parser = ArgumentParser(description="Self-contained GSIR PBR Viewer (Directional Light, No Fallback)")
    pipeline = PipelineParams(parser)

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to GSIR checkpoint")
    parser.add_argument("--bg", type=float, default=0.0, help="Background gray [0,1]")
    parser.add_argument("--sh", type=int, default=3, help="SH degree for GaussianModel")

    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)

    parser.add_argument("--eye", type=float, nargs=3, default=[0.0, 0.0, 3.0])
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, 1.0, 0.0])

    # PBR light + tone/gamma toggles
    parser.add_argument("--light_dir", type=float, nargs=3, default=[0.3, 0.6, 0.7], help="Directional light (world) xyz")
    parser.add_argument("--light_intensity", type=float, default=3.0, help="Directional light intensity (scalar)")
    parser.add_argument("--tone", action="store_true", help="Enable tone mapping (ACES-ish)")
    parser.add_argument("--gamma", action="store_true", help="Enable gamma 2.2")
    parser.add_argument("--metallic", action="store_true", help="Use predicted metallic map")

    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load gaussians
    gaussians = GaussianModel(args.sh)
    ckpt = torch.load(args.checkpoint, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(ckpt, tuple):
        model_params = ckpt[0]
    elif isinstance(ckpt, dict):
        model_params = ckpt.get("gaussians", ckpt.get("state_dict", ckpt))
    else:
        raise TypeError("Unsupported checkpoint format for GSIR checkpoint.")
    gaussians.restore(model_params)

    # Camera
    eye = torch.tensor(args.eye, dtype=torch.float32, device=device)
    center = torch.tensor(args.center, dtype=torch.float32, device=device)
    up = torch.tensor(args.up, dtype=torch.float32, device=device)
    cam = build_minicam(
        width=args.width, height=args.height, fov_deg=args.fov_deg,
        eye=eye, center=center, up=up, device=device, znear=args.znear, zfar=args.zfar,
    )

    app = App(
        cam=cam, gaussians=gaussians, pipeline=pipeline.extract(args),
        bg=args.bg,
        light_dir=tuple(args.light_dir),
        light_intensity=args.light_intensity,
        tone=args.tone, gamma=args.gamma,
        use_metallic=args.metallic,
    )
    app.run()

if __name__ == "__main__":
    main()
