from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Tuple

from OpenGL import GL as gl


class OpenGLTexture2D:
    """Simple OpenGL texture wrapper for uploading numpy images."""

    def __init__(self) -> None:
        self.texture_id: Optional[int] = None
        self.width: int = 0
        self.height: int = 0
        self.channels: int = 0
        self._pixel_format: Optional[int] = None
        self._pixel_type: Optional[int] = None
        self._internal_format: Optional[int] = None

    @property
    def is_initialized(self) -> bool:
        return self.texture_id is not None

    def release(self) -> None:
        if gl is not None and self.texture_id is not None:
            gl.glDeleteTextures(int(self.texture_id))
            self.texture_id = None

    def update(self, image: np.ndarray) -> None:
        if gl is None:
            raise RuntimeError("OpenGL.GL is not available to upload textures.")
        if image.ndim != 3:
            raise ValueError("Expected image with shape (H, W, C).")

        data = np.ascontiguousarray(image)
        height, width, channels = data.shape

        if channels == 3:
            pixel_format = gl.GL_RGB
        elif channels == 4:
            pixel_format = gl.GL_RGBA
        else:
            raise ValueError(f"Unsupported channel count: {channels}.")

        if data.dtype == np.uint8:
            pixel_type = gl.GL_UNSIGNED_BYTE
            internal_format = pixel_format
        elif data.dtype == np.float32:
            pixel_type = gl.GL_FLOAT
            internal_format = gl.GL_RGB32F if channels == 3 else gl.GL_RGBA32F
        else:
            raise TypeError(f"Unsupported dtype for texture upload: {data.dtype}.")

        if self.texture_id is None:
            self.texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        else:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        needs_reallocate = (
            width != self.width
            or height != self.height
            or pixel_format != self._pixel_format
            or pixel_type != self._pixel_type
            or internal_format != self._internal_format
        )

        if needs_reallocate:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                internal_format,
                width,
                height,
                0,
                pixel_format,
                pixel_type,
                data,
            )
            self.width = width
            self.height = height
            self.channels = channels
            self._pixel_format = pixel_format
            self._pixel_type = pixel_type
            self._internal_format = internal_format
        else:
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D,
                0,
                0,
                0,
                width,
                height,
                pixel_format,
                pixel_type,
                data,
            )

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def __del__(self):  # pragma: no cover - cleanup guard
        try:
            self.release()
        except Exception:
            pass


def _infer_formats_from_tensor(tensor: torch.Tensor) -> Tuple[int, int, int]:
    channels = tensor.shape[-1]
    if channels == 3:
        pixel_format = gl.GL_RGB
        internal_float = gl.GL_RGB32F
        internal_uint8 = gl.GL_RGB8
    elif channels == 4:
        pixel_format = gl.GL_RGBA
        internal_float = gl.GL_RGBA32F
        internal_uint8 = gl.GL_RGBA8
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    if tensor.dtype == torch.float32:
        return internal_float, pixel_format, gl.GL_FLOAT
    if tensor.dtype == torch.uint8:
        return internal_uint8, pixel_format, gl.GL_UNSIGNED_BYTE
    raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}")


class CpuTextureBackend:
    """Uploads data via the original OpenGLTexture2D helper."""

    def __init__(self) -> None:
        self.texture = OpenGLTexture2D()

    def upload(self, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu":
            tensor = tensor.detach().to("cpu")
        else:
            tensor = tensor.detach()
        array = np.ascontiguousarray(tensor.numpy())
        self.texture.update(array)

    @property
    def texture_id(self) -> Optional[int]:
        return self.texture.texture_id

    def release(self) -> None:
        self.texture.release()


try:
    import pycuda.autoprimaryctx  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gl as cudagl

    _PYCUDA_AVAILABLE = True
    _PYCUDA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime environment dependent
    cuda = None
    cudagl = None
    _PYCUDA_AVAILABLE = False
    _PYCUDA_IMPORT_ERROR = exc


class CudaTexture2D(OpenGLTexture2D):
    """Extends the OpenGL texture with CUDA-OpenGL interop uploads."""

    def __init__(self) -> None:
        if not _PYCUDA_AVAILABLE:
            raise RuntimeError(f"PyCUDA is not available: {_PYCUDA_IMPORT_ERROR}")
        super().__init__()
        self._registered_mapping: Optional[cudagl.RegisteredImage] = None
        self._interop_ready = False

    def release(self) -> None:
        if self._registered_mapping is not None:
            self._registered_mapping.unregister()
            self._registered_mapping = None
        super().release()

    def _ensure_interop(self) -> None:
        """
        Ensures the CUDA context is initialized.
        REMOVED: cuda.init() and pycuda.gl.autoinit.
        PyTorch will initialize a CUDA context when the first CUDA tensor is created.
        We will rely on that context being active. PyCUDA operations will
        automatically use the existing PyTorch context.
        """
        if self._interop_ready:
            return

        # This check is now implicit. The 'upload_cuda' method ensures the tensor is
        # on a CUDA device, which means PyTorch has already created a context.
        # If no CUDA device is available or PyTorch fails, it will raise its own error.
        self._interop_ready = True

    def _allocate_for_tensor(self, tensor: torch.Tensor) -> None:
        # DEBUG: Check if an error state already exists
        existing_error = gl.glGetError()
        if existing_error != gl.GL_NO_ERROR:
            print(f"!!! OpenGL error existed BEFORE this function: {existing_error}")
        self._ensure_interop()
        internal_format, pixel_format, pixel_type = _infer_formats_from_tensor(tensor)
        height, width, _ = tensor.shape

        if self.texture_id is None:
            self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            internal_format,
            width,
            height,
            0,
            pixel_format,
            pixel_type,
            None,
        )
        # gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        if self._registered_mapping is not None:
            self._registered_mapping.unregister()
        self._registered_mapping = cudagl.RegisteredImage(
            int(self.texture_id),
            gl.GL_TEXTURE_2D,
            cudagl.graphics_map_flags.WRITE_DISCARD,
        )

        self.width = width
        self.height = height
        self.channels = tensor.shape[-1]
        self._pixel_format = pixel_format
        self._pixel_type = pixel_type
        self._internal_format = internal_format
        self._dtype = tensor.dtype

    def upload_cuda(self, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cuda":
            raise RuntimeError("CudaTexture2D expects a CUDA tensor.")
        tensor = tensor.contiguous()
        if (
            self.texture_id is None
            or tensor.shape[0] != self.height
            or tensor.shape[1] != self.width
            or tensor.shape[2] != self.channels
            or tensor.dtype != getattr(self, "_dtype", None)
        ):
            self._allocate_for_tensor(tensor)

        if self._registered_mapping is None:
            raise RuntimeError("CUDA texture not registered for interop.")

        mapping = self._registered_mapping.map()
        cuda_array = mapping.array(0, 0)
        copy = cuda.Memcpy2D()
        copy.set_src_device(int(tensor.data_ptr()))
        copy.set_dst_array(cuda_array)
        row_bytes = tensor.shape[1] * tensor.shape[2] * tensor.element_size()
        copy.src_pitch = row_bytes
        copy.dst_pitch = row_bytes
        copy.width_in_bytes = row_bytes
        copy.height = tensor.shape[0]
        # torch.cuda.synchronize()
        copy(aligned=True)
        mapping.unmap()


class CudaTextureBackend:
    def __init__(self) -> None:
        self.texture = CudaTexture2D()

    def upload(self, tensor: torch.Tensor) -> None:
        self.texture.upload_cuda(tensor)

    @property
    def texture_id(self) -> Optional[int]:
        return self.texture.texture_id

    def release(self) -> None:
        self.texture.release()


def create_texture_backend():
    """Returns (backend, warning_message)."""
    if _PYCUDA_AVAILABLE:
        try:
            return CudaTextureBackend(), None
        except Exception as exc:
            warning = f"[viewer] CUDA texture failed ({exc}). Falling back to CPU uploads."
            print(warning)
            return CpuTextureBackend(), warning
    warning = None
    if "_PYCUDA_IMPORT_ERROR" in globals() and _PYCUDA_IMPORT_ERROR is not None:
        warning = f"[viewer] PyCUDA unavailable ({_PYCUDA_IMPORT_ERROR}). Using CPU texture uploads."
        print(warning)
    return CpuTextureBackend(), warning
