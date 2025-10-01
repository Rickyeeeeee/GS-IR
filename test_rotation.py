#!/usr/bin/env python3
import math
import torch

from utils.general_utils import build_rotation
from utils.general_utils import rotation_to_quaternion

def quat_from_axis_angle_wxyz(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half = 0.5 * angle
    s = torch.sin(half)[..., None]
    c = torch.cos(half)[..., None]
    return torch.cat([c, axis * s], dim=-1)  # [w, x, y, z]

def embed_to_4x4(R: torch.Tensor) -> torch.Tensor:
    """Embed (...,3,3) into (...,4,4) homogeneous matrices."""
    eye = torch.eye(4, dtype=R.dtype, device=R.device).expand(*R.shape[:-2], 4, 4).clone()
    eye[..., :3, :3] = R
    return eye

# ---------------- Tests ----------------
def main():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print(f"Using device={device}, dtype={dtype}")

    # 1) Random batch round-trip q -> R -> q
    N = 1000
    q = torch.randn(N, 4, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    R = build_rotation(q)
    q2 = rotation_to_quaternion(R)

    # fix global sign ambiguity: make dot(q2,q) >= 0
    flip = (q2 * q).sum(dim=-1, keepdim=True) < 0
    q2 = torch.where(flip, -q2, q2)

    max_q_err = (q2 - q).abs().max().item()
    print(f"[Random] max |q2 - q| = {max_q_err:.3e}")
    assert torch.allclose(q2, q, atol=1e-5, rtol=1e-5), "Random round-trip q->R->q failed"

    # R consistency: R from q equals R from q2
    R2 = build_rotation(q2)
    max_R_err = (R2 - R).abs().max().item()
    print(f"[Random] max |R2 - R| = {max_R_err:.3e}")
    assert torch.allclose(R2, R, atol=1e-5, rtol=1e-5), "R mismatch after round-trip"

    # 2) Known angles / axes
    axes = torch.tensor([[1.,0.,0.],
                         [0.,1.,0.],
                         [0.,0.,1.],
                         [1.,2.,3.]], device=device, dtype=dtype)
    angles = torch.tensor([0.0, math.pi/2, math.pi, math.pi/3], device=device, dtype=dtype)  # 0°, 90°, 180°, 60°
    qk = quat_from_axis_angle_wxyz(axes, angles)
    Rk = build_rotation(qk)
    qk2 = rotation_to_quaternion(Rk)
    flipk = (qk2 * qk).sum(dim=-1, keepdim=True) < 0
    qk2 = torch.where(flipk, -qk2, qk2)

    max_qk_err = (qk2 - qk).abs().max().item()
    print(f"[Known]  max |qk2 - qk| = {max_qk_err:.3e}")
    assert torch.allclose(qk2, qk, atol=1e-6, rtol=1e-6), "Known-angle round-trip failed"

    # 3) 4x4 input support
    R4 = embed_to_4x4(Rk)
    q4 = rotation_to_quaternion(R4)
    flip4 = (q4 * qk).sum(dim=-1, keepdim=True) < 0
    q4 = torch.where(flip4, -q4, q4)
    max_q4_err = (q4 - qk).abs().max().item()
    print(f"[4x4]    max |q4 - qk| = {max_q4_err:.3e}")
    assert torch.allclose(q4, qk, atol=1e-6, rtol=1e-6), "4x4 input round-trip failed"

    print("✅ All quaternion/rotation tests passed.")

if __name__ == "__main__":
    main()