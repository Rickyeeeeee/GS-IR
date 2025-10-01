#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import random
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image: Image.Image, resolution: Sequence[int]) -> torch.Tensor:
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
) -> Callable:
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    return strip_lowerdiag(sym)


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@torch.no_grad()
def rotation_to_quaternion(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert rotation matrices to unit quaternions in [w, x, y, z] (scalar first).
    R: (...,3,3) or (...,4,4)
    Returns: (...,4) with [w, x, y, z]
    """
    if R.shape[-2:] == (4, 4):
        R = R[..., :3, :3]

    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    t = m00 + m11 + m22

    # Allocate
    shp = R.shape[:-2]
    w = torch.empty(shp, dtype=R.dtype, device=R.device)
    x = torch.empty_like(w)
    y = torch.empty_like(w)
    z = torch.empty_like(w)

    # Case 1: positive trace
    c0 = t > 0
    s0 = torch.sqrt(torch.clamp(t[c0] + 1.0, min=eps)) * 2.0  # s = 4*w
    w[c0] = 0.25 * s0
    x[c0] = (m21[c0] - m12[c0]) / s0
    y[c0] = (m02[c0] - m20[c0]) / s0
    z[c0] = (m10[c0] - m01[c0]) / s0

    # Case 2: m00 is largest
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    s1 = torch.sqrt(torch.clamp(1.0 + m00[c1] - m11[c1] - m22[c1], min=eps)) * 2.0  # s = 4*x
    w[c1] = (m21[c1] - m12[c1]) / s1
    x[c1] = 0.25 * s1
    y[c1] = (m01[c1] + m10[c1]) / s1
    z[c1] = (m02[c1] + m20[c1]) / s1

    # Case 3: m11 is largest
    c2 = (~c0) & (~c1) & (m11 > m22)
    s2 = torch.sqrt(torch.clamp(1.0 - m00[c2] + m11[c2] - m22[c2], min=eps)) * 2.0  # s = 4*y
    w[c2] = (m02[c2] - m20[c2]) / s2
    x[c2] = (m01[c2] + m10[c2]) / s2
    y[c2] = 0.25 * s2
    z[c2] = (m12[c2] + m21[c2]) / s2

    # Case 4: m22 is largest
    c3 = ~(c0 | c1 | c2)
    s3 = torch.sqrt(torch.clamp(1.0 - m00[c3] - m11[c3] + m22[c3], min=eps)) * 2.0  # s = 4*z
    w[c3] = (m10[c3] - m01[c3]) / s3
    x[c3] = (m02[c3] + m20[c3]) / s3
    y[c3] = (m12[c3] + m21[c3]) / s3
    z[c3] = 0.25 * s3

    q = torch.stack([w, x, y, z], dim=-1)
    # Normalize to be safe
    q = q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))
    return q


def euler_to_matrix(euler: torch.Tensor,
                    order: str = "ZYX",
                    degrees: bool = False,
                    intrinsic: bool = True) -> torch.Tensor:
    """
    Convert Euler angles to a rotation matrix.

    Args:
        euler: (..., 3) tensor of angles [a1, a2, a3].
        order: Axis order string from {"XYZ","XZY","YXZ","YZX","ZXY","ZYX"} (case-insensitive).
               Example: "ZYX" (yaw Z, pitch Y, roll X).
        degrees: If True, interpret angles in degrees; otherwise radians.
        intrinsic: If True, use intrinsic (rotating/body axes) composition:
                   R = R(axis1,a1) @ R(axis2,a2) @ R(axis3,a3).
                   If False, use extrinsic (fixed/world axes), which is the reverse order:
                   R = R(axis3,a3) @ R(axis2,a2) @ R(axis1,a1).

    Returns:
        (..., 3, 3) rotation matrices with the same dtype/device as `euler`.
    """
    order = order.upper()
    if len(order) != 3 or any(c not in "XYZ" for c in order) or len(set(order)) != 3:
        raise ValueError("order must be a 3-letter Tait–Bryan sequence using X,Y,Z exactly once (e.g., 'ZYX').")

    a1, a2, a3 = euler.unbind(dim=-1)
    if degrees:
        a1 = torch.deg2rad(a1)
        a2 = torch.deg2rad(a2)
        a3 = torch.deg2rad(a3)

    def Rx(a):
        ca, sa = torch.cos(a), torch.sin(a)
        R = torch.zeros(a.shape + (3, 3), dtype=euler.dtype, device=euler.device)
        R[..., 0, 0] = 1
        R[..., 1, 1] = ca; R[..., 1, 2] = -sa
        R[..., 2, 1] = sa; R[..., 2, 2] =  ca
        return R

    def Ry(a):
        ca, sa = torch.cos(a), torch.sin(a)
        R = torch.zeros(a.shape + (3, 3), dtype=euler.dtype, device=euler.device)
        R[..., 1, 1] = 1
        R[..., 0, 0] =  ca; R[..., 0, 2] =  sa
        R[..., 2, 0] = -sa; R[..., 2, 2] =  ca
        return R

    def Rz(a):
        ca, sa = torch.cos(a), torch.sin(a)
        R = torch.zeros(a.shape + (3, 3), dtype=euler.dtype, device=euler.device)
        R[..., 2, 2] = 1
        R[..., 0, 0] =  ca; R[..., 0, 1] = -sa
        R[..., 1, 0] =  sa; R[..., 1, 1] =  ca
        return R

    Ax = {"X": Rx, "Y": Ry, "Z": Rz}
    R1 = Ax[order[0]](a1)
    R2 = Ax[order[1]](a2)
    R3 = Ax[order[2]](a3)

    # Intrinsic (body): R = R1 @ R2 @ R3
    # Extrinsic (world): apply in reverse order: R = R3 @ R2 @ R1
    return R1 @ R2 @ R3 if intrinsic else R3 @ R2 @ R1


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent: bool, seed: int = 0) -> None:
    # old_f = sys.stdout

    # class F:
    #     def __init__(self, silent):
    #         self.silent = silent

    #     def write(self, x: str) -> None:
    #         if not self.silent:
    #             if x.endswith("\n"):
    #                 old_f.write(
    #                     x.replace("\n", f" [{str(datetime.now().strftime('%d/%m %H:%M:%S'))}]\n")
    #                 )
    #             else:
    #                 old_f.write(x)

    #     def flush(self) -> None:
    #         old_f.flush()

    # sys.stdout = F(silent)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))
