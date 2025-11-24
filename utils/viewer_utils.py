import torch
import os
import math
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from utils.general_utils import euler_to_matrix


def get_canonical_rays(H: int, W: int, tan_fovx: float, tan_fovy: float) -> torch.Tensor:
    cen_x = W / 2
    cen_y = H / 2
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    x, y = torch.meshgrid(
        torch.arange(W),
        torch.arange(H),
        indexing="xy",
    )
    x = x.flatten()  # [H * W]
    y = y.flatten()  # [H * W]
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cen_x + 0.5) / focal_x,
                (y - cen_y + 0.5) / focal_y,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [H * W, 3]
    # NOTE: it is not normalized
    return camera_dirs.cuda()

def tensor_to_raw_rgba(img: torch.Tensor) -> np.ndarray:
    """
    img: torch float tensor [H,W,3] in [0,1] (CUDA or CPU).
    returns contiguous NumPy float32 [H,W,4] with alpha=1.
    """
    img = img.clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)  # [H,W,3]
    H, W, _ = img.shape
    if img.flags['C_CONTIGUOUS'] is False:
        img = np.ascontiguousarray(img)
    alpha = np.ones((H, W, 1), dtype=np.float32)
    rgba = np.concatenate([img, alpha], axis=-1)  # [H,W,4]
    return np.ascontiguousarray(rgba)


# ---- simple tone/gamma helpers for env background ----
def _aces_film(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    a = 0.055
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + a) * torch.pow(x, 1 / 2.4) - a)


def _sample_env_latlong(latlong_map: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Sample a latlong HDRI with per-pixel ray directions.
    Args:
        latlong_map: [H_env, W_env, 3] float tensor (CUDA)
        dirs: [H, W, 3] normalized ray directions in world space
    Returns:
        [H, W, 3] sampled color (linear)
    """
    rotation_matrix = torch.tensor(
        euler_to_matrix(
           torch.deg2rad(torch.tensor(180.0)), 
           torch.deg2rad(torch.tensor(0.0)), 
           torch.deg2rad(torch.tensor(0.0)) 
        ), dtype=dirs.dtype, device=dirs.device)

    dirs = torch.matmul(dirs, rotation_matrix)
    v = torch.nn.functional.normalize(dirs, p=2, dim=-1)
    tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
    tv = torch.acos(torch.clamp(v[..., 1:2], min=-1.0, max=1.0)) / np.pi
    texcoord = torch.cat((tu, tv), dim=-1)
    sampled = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode="linear")[0]
    return sampled


def _sample_env_cubemap(cubemap: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Sample a cubemap (6, H, W, C) using normalized directions."""
    dirs = torch.nn.functional.normalize(dirs, p=2, dim=-1)
    x, y, z = dirs.unbind(dim=-1)
    abs_x, abs_y, abs_z = torch.abs(x), torch.abs(y), torch.abs(z)
    max_axis = torch.argmax(torch.stack((abs_x, abs_y, abs_z), dim=-1), dim=-1)

    face = torch.empty_like(max_axis)
    face[max_axis == 0] = torch.where(x[max_axis == 0] >= 0, 0, 1)
    face[max_axis == 1] = torch.where(y[max_axis == 1] >= 0, 2, 3)
    face[max_axis == 2] = torch.where(z[max_axis == 2] >= 0, 4, 5)

    eps = 1e-6
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    mask = face == 0  # +X
    if mask.any():
        ax = torch.clamp(abs_x[mask], min=eps)
        u[mask] = -z[mask] / ax
        v[mask] = -y[mask] / ax

    mask = face == 1  # -X
    if mask.any():
        ax = torch.clamp(abs_x[mask], min=eps)
        u[mask] = z[mask] / ax
        v[mask] = -y[mask] / ax

    mask = face == 2  # +Y
    if mask.any():
        ay = torch.clamp(abs_y[mask], min=eps)
        u[mask] = x[mask] / ay
        v[mask] = z[mask] / ay

    mask = face == 3  # -Y
    if mask.any():
        ay = torch.clamp(abs_y[mask], min=eps)
        u[mask] = x[mask] / ay
        v[mask] = -z[mask] / ay

    mask = face == 4  # +Z
    if mask.any():
        az = torch.clamp(abs_z[mask], min=eps)
        u[mask] = x[mask] / az
        v[mask] = -y[mask] / az

    mask = face == 5  # -Z
    if mask.any():
        az = torch.clamp(abs_z[mask], min=eps)
        u[mask] = -x[mask] / az
        v[mask] = -y[mask] / az

    grid = torch.stack((u, v), dim=-1)
    out = torch.zeros((*dirs.shape[:2], cubemap.shape[-1]), device=cubemap.device, dtype=cubemap.dtype)
    grid_4d = grid.unsqueeze(0)  # [1, H, W, 2] for grid_sample

    for face_idx in range(6):
        face_mask = face == face_idx
        if not torch.any(face_mask):
            continue
        tex = cubemap[face_idx].permute(2, 0, 1).unsqueeze(0)  # [1, C, Hc, Wc]
        sampled = F.grid_sample(
            tex,
            grid_4d,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0].permute(1, 2, 0)
        out[face_mask] = sampled[face_mask]

    return out

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


def cubemap_to_latlong(cubemap: torch.Tensor, res_hw: List[int]) -> torch.Tensor:
    """Convert a cubemap (6, H, W, C) into a latlong map [H_out, W_out, C]."""
    H_out, W_out = res_hw
    gy, gx = torch.meshgrid(
        torch.linspace(0.0, 1.0, H_out, device=cubemap.device, dtype=cubemap.dtype, requires_grad=False),
        torch.linspace(0.0, 1.0, W_out, device=cubemap.device, dtype=cubemap.dtype, requires_grad=False),
        indexing="ij",
    )
    alpha = (gx - 0.5) * (2.0 * np.pi)
    theta = gy * np.pi
    sin_theta = torch.sin(theta)
    dirs = torch.stack(
        (
            sin_theta * torch.sin(alpha),
            torch.cos(theta),
            -sin_theta * torch.cos(alpha),
        ),
        dim=-1,
    )
    return _sample_env_cubemap(cubemap, dirs)

def tensor_to_dpg_rgba(img: torch.Tensor) -> np.ndarray:
    """Convert [H,W,3] float tensor in [0,1] to flattened RGBA float array for DearPyGui."""
    img = img.clamp(0.0, 1.0)
    H, W, _ = img.shape
    alpha = torch.ones(H, W, 1, device=img.device, dtype=img.dtype)
    rgba = torch.cat([img, alpha], dim=-1).contiguous()  # [H,W,4]
    return rgba.detach().cpu().numpy().astype(np.float32).ravel()

def euler_to_matrix(yaw: float, pitch: float, roll: float, order="zyx") -> np.ndarray:
    """
    Build a rotation matrix from Euler angles.
    yaw   = rotation around Y axis
    pitch = rotation around X axis
    roll  = rotation around Z axis
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    R_yaw = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy],
    ], dtype=np.float32)

    R_pitch = np.array([
        [1,  0,   0],
        [0, cp, -sp],
        [0, sp,  cp],
    ], dtype=np.float32)

    R_roll = np.array([
        [cr, -sr, 0],
        [sr,  cr, 0],
        [ 0,   0, 1],
    ], dtype=np.float32)

    if order == "zyx":
        return R_roll @ R_yaw @ R_pitch
    elif order == "xyz":
        return R_pitch @ R_yaw @ R_roll
    else:
        raise ValueError("Unsupported order")
