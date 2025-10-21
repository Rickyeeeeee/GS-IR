#!/usr/bin/env python3
"""
Simple sanity check for uploading a torch tensor into an OpenGL texture.

The script creates a hidden GLFW window to obtain an OpenGL context, uploads a
test tensor through the viewer.gl_utils backends, reads the texture back, and
reports the maximum difference between the original tensor and the texture
contents. Helpful for debugging CUDA-OpenGL interop issues.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from OpenGL import GL as gl

from viewer.gl_utils import CpuTextureBackend, CudaTextureBackend, _infer_formats_from_tensor


@dataclass
class TestResult:
    max_abs_diff: float
    dtype: torch.dtype
    device: torch.device
    shape: Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Device for the test tensor.")
    parser.add_argument("--dtype", choices=("float32", "uint8"), default="float32", help="Tensor dtype to test.")
    parser.add_argument("--width", type=int, default=128, help="Texture width.")
    parser.add_argument("--height", type=int, default=128, help="Texture height.")
    parser.add_argument("--channels", type=int, default=4, help="Number of channels to upload (3 or 4).")
    return parser.parse_args()


@contextlib.contextmanager
def hidden_glfw_context(width: int, height: int):
    """Create a hidden GLFW window to host the GL context."""
    try:
        import glfw  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("glfw is required for this test script.") from exc

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW.")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)

    window = glfw.create_window(width, height, "Tensor->GL Interop Test", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window.")

    try:
        glfw.make_context_current(window)
        yield window
    finally:
        glfw.make_context_current(None)
        glfw.destroy_window(window)
        glfw.terminate()


def make_test_tensor(
    height: int, width: int, channels: int, dtype_name: str, device_name: str
) -> torch.Tensor:
    """Create a deterministic tensor for upload."""
    if channels not in (3, 4):
        raise ValueError("channels must be 3 or 4.")

    base = torch.arange(height * width * channels, dtype=torch.float32).reshape(height, width, channels)

    if dtype_name == "uint8":
        tensor = (base % 256).to(torch.uint8)
    else:
        tensor = (base / base.max()).to(torch.float32)

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return tensor.contiguous().to(device)
    return tensor.contiguous()


def upload_tensor(tensor: torch.Tensor):
    """Upload the tensor into an OpenGL texture and return (backend, texture id)."""
    if tensor.device.type == "cuda":
        backend = CudaTextureBackend()
    else:
        backend = CpuTextureBackend()

    backend.upload(tensor)
    texture_id = backend.texture_id
    if texture_id is None:
        backend.release()
        raise RuntimeError("Texture upload failed: texture id is None.")

    # The caller is responsible for cleaning up the backend/texture after reading.
    return backend, texture_id


def read_back_texture(tensor: torch.Tensor, texture_id: int) -> np.ndarray:
    """Read back the texture into a numpy array matching the tensor dtype."""
    _, pixel_format, pixel_type = _infer_formats_from_tensor(tensor.detach().to("cpu"))
    height, width, channels = tensor.shape

    if tensor.dtype == torch.uint8:
        buffer = np.empty((height, width, channels), dtype=np.uint8)
    else:
        buffer = np.empty((height, width, channels), dtype=np.float32)

    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, pixel_format, pixel_type, buffer)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return buffer


def compare_tensor_and_texture(tensor: torch.Tensor, texture_np: np.ndarray) -> float:
    """Return the maximum absolute difference between tensor and texture."""
    reference = tensor.detach().to("cpu").numpy()
    if tensor.dtype == torch.uint8:
        diff = np.abs(reference.astype(np.int16) - texture_np.astype(np.int16)).max()
    else:
        diff = np.abs(reference - texture_np).max()
    return float(diff)


def run_test(args: argparse.Namespace) -> TestResult:
    tensor = make_test_tensor(args.height, args.width, args.channels, args.dtype, args.device)
    backend, texture_id = upload_tensor(tensor)
    try:
        texture_np = read_back_texture(tensor, texture_id)
        diff = compare_tensor_and_texture(tensor, texture_np)
    finally:
        gl.glDeleteTextures(int(texture_id))
        backend.release()
    return TestResult(diff, tensor.dtype, tensor.device, tensor.shape)  # type: ignore[arg-type]


def main() -> int:
    args = parse_args()

    with hidden_glfw_context(args.width, args.height):
        try:
            result = run_test(args)
        except RuntimeError as exc:
            print(f"[interop-test] Error: {exc}")
            return 1

    print("[interop-test] Success!")
    print(f"  device : {result.device}")
    print(f"  dtype  : {result.dtype}")
    print(f"  shape  : {result.shape}")
    print(f"  max |tensor - texture| : {result.max_abs_diff:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
