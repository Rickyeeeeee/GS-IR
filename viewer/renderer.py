from __future__ import annotations

import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from gaussian_renderer import render
from pbr import pbr_shading
from utils.viewer_utils import get_canonical_rays

from .gl_utils import CpuTextureBackend, CudaTextureBackend, create_texture_backend
from .state import ViewerState


class ViewerRenderer:
    """Runs the rendering pipeline and uploads the result to an OpenGL texture."""

    def __init__(self, state: ViewerState):
        self.state = state
        self.render_image_tensor: Optional[torch.Tensor] = None
        self.texture_backend, self.backend_warning = create_texture_backend()
        self.last_render_error: Optional[str] = None
        self.image_display_time_ms = 0.0
        self._image_dirty = False
        self.has_image = False
        self._last_uploaded_shape: Optional[Tuple[int, int, int]] = None
        self.resolution_scale: float = 1.0

    def render_current(self) -> torch.Tensor:
        state = self.state
        t_start = time.perf_counter()
        gaussians = state.prepare_render_gaussians()
        t_after_prepare = time.perf_counter()
        
        with torch.no_grad():

            view = state.camera
            rendering_result = render(
                viewpoint_camera=view,
                pc=gaussians,
                pipe=state.args.pipeline,
                bg_color=state.background,
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
            canonical_rays = get_canonical_rays(H, W, view.FoVx, view.FoVy)
            view_dirs = -(
                (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3]).sum(dim=-1).reshape(H, W, 3)
            )

            env_yaw_rad = torch.tensor(state.hdri_rotation_deg * (torch.pi / 180.0), device=view_dirs.device)
            env_cos_yaw = torch.cos(env_yaw_rad)
            env_sin_yaw = torch.sin(env_yaw_rad)
            x, y, z = view_dirs[..., 0], view_dirs[..., 1], view_dirs[..., 2]
            x_rotated = x * env_cos_yaw - z * env_sin_yaw
            z_rotated = x * env_sin_yaw + z * env_cos_yaw
            light_sample_dirs = torch.stack([x_rotated, y, z_rotated], dim=-1)

            result = pbr_shading(
                light=state.light,
                normals=normal_map.permute(1, 2, 0),
                view_dirs=light_sample_dirs,
                mask=rendering_result["normal_mask"].permute(1, 2, 0),
                albedo=albedo_map.permute(1, 2, 0),
                roughness=roughness_map.permute(1, 2, 0),
                metallic=metallic_map.permute(1, 2, 0) if state.enable_metallic else None,
                tone=state.enable_tone,
                gamma=state.enable_gamma,
                brdf_lut=state.brdf_lut,
            )

            render_rgb = result["render_rgb"].clamp(0.0, 1.0)
            t_after_shading = time.perf_counter()

            composed_rgb = state.composite_render(
                render_rgb,
                opacity_mask,
                light_sample_dirs,
                state.enable_tone,
                state.enable_gamma,
            )
            t_after_env = time.perf_counter()

        state.profile_timings = {
            "prepare": (t_after_prepare - t_start) * 1000.0,
            "render": (t_after_render - t_after_prepare) * 1000.0,
            "shading": (t_after_shading - t_after_render) * 1000.0,
            "composite": (t_after_env - t_after_shading) * 1000.0,
            "total": (t_after_env - t_start) * 1000.0,
            "gaussian_count": float(state.render_gaussians.get_xyz.shape[0]),
        }

        return composed_rgb

    def update_render_buffer(self):
        # try:
        img = self.render_current().contiguous()
        self.render_image_tensor = img
        self.has_image = True
        self._image_dirty = True
        self.last_render_error = None
        # except Exception as exc:  # pragma: no cover - visualization helper path
        #     self.last_render_error = str(exc)
        #     self.has_image = False

    def ensure_camera_matches_size(self, width: int, height: int) -> None:
        self.state.update_camera_resolution(width, height)

    def ensure_render_texture(self) -> Tuple[Optional[int], float, Optional[str]]:
        if not self.has_image or self.render_image_tensor is None:
            return None, 0.0, None

        upload_ms = 0.0
        if self._image_dirty:
            t_start = time.perf_counter()
            tensor = (self.render_image_tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            if tensor.shape[-1] == 3:
                alpha = torch.full_like(tensor[..., :1], 255)
                tensor = torch.cat([tensor, alpha], dim=-1)
            tensor = tensor.contiguous()
            current_shape = tuple(int(dim) for dim in tensor.shape)
            if self._last_uploaded_shape != current_shape:
                self.texture_backend.release()
                self._last_uploaded_shape = None

            if isinstance(self.texture_backend, CudaTextureBackend):
                self.texture_backend.upload(tensor.to(self.state.device))
            else:
                self.texture_backend.upload(tensor.to("cpu"))
            self._image_dirty = False
            self._last_uploaded_shape = current_shape
            upload_ms = (time.perf_counter() - t_start) * 1000.0

        texture_id = self.texture_backend.texture_id
        return (int(texture_id) if texture_id is not None else None), upload_ms, None

    def set_resolution_scale(self, scale: float) -> None:
        clamped = max(0.1, min(1.0, float(scale)))
        if abs(clamped - self.resolution_scale) < 1e-4:
            return
        self.resolution_scale = clamped
        self.texture_backend.release()
        self._last_uploaded_shape = None
        self._image_dirty = True
