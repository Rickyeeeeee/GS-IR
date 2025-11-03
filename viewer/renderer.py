from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import _C
from gaussian_renderer import render
from pbr import pbr_shading
from pbr.shade import aces_film as pbr_aces_film, linear_to_srgb as pbr_linear_to_srgb
from utils.graphics_utils import getProjectionMatrix
from utils.viewer_utils import get_canonical_rays

from .gl_utils import CpuTextureBackend, CudaTextureBackend, create_texture_backend
from .state import ViewerState

def saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)


def DistributionGGX(
    normals: torch.Tensor,  # [H, W, 3]
    half_dirs: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
) -> torch.Tensor:
    a = roughness * roughness
    a2 = a * a
    NoH = saturate_dot(normals, half_dirs)
    NoH2 = NoH * NoH

    nom = a2
    denom = (NoH2 * (a2 - 1.0) + 1.0)
    denom = np.pi * denom * denom

    return nom / denom


def GeometrySchlickGGX(
    NoV: torch.Tensor, # [H, W, 1]
    roughness: torch.Tensor,  # [H, W, 1]
) -> torch.Tensor:
    r = roughness + 1.0
    k = (r * r) / 8.0
    nom = NoV
    denom = NoV * (1.0 - k) + k

    return nom / denom

def GeometrySmith(
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    light_dirs: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
) -> torch.Tensor:
    NoV = saturate_dot(normals, view_dirs)
    NoL = saturate_dot(normals, light_dirs)
    ggx2 = GeometrySchlickGGX(NoV, roughness)
    ggx1 = GeometrySchlickGGX(NoL, roughness)

    return ggx1 * ggx2


def fresnelSchlick(
    HoV: torch.Tensor,  # [H, W, 1]
    F0: torch.Tensor,  # [H, W, 3]
) -> torch.Tensor:
    return F0 + (1.0 - F0) * torch.pow((1.0 - HoV).clamp(0.0, 1.0), 5)


def light_pbr_shading(
    light_position: torch.Tensor,  # [3]
    light_intensity: torch.Tensor,  # [3]
    points: torch.Tensor,  # [H, W, 3]
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    albedo: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
    mask: torch.Tensor,  # [H, W, 1]
    linear: bool = False,
    metallic: Optional[torch.Tensor] = None,
    shadow: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if background is None:
        background = torch.zeros_like(normals)

    light_dirs = F.normalize(light_position - points, p=2, dim=-1)
    half_dirs = (light_dirs + view_dirs) / 2.0
    distance = torch.norm(light_position - points, p=2, dim=-1, keepdim=True)
    attenuation = 1.0 / torch.pow(distance, 2)
    radiance = light_intensity * attenuation

    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic

    NoV = saturate_dot(normals, view_dirs)
    NoL = saturate_dot(normals, light_dirs)
    HoV = saturate_dot(half_dirs, view_dirs)
    NDF = DistributionGGX(normals=normals, half_dirs=half_dirs, roughness=roughness)
    G = GeometrySmith(normals=normals, view_dirs=view_dirs, light_dirs=light_dirs, roughness=roughness)
    fresnel = fresnelSchlick(HoV=HoV, F0=F0)

    numerator = NDF * G * fresnel
    denominator = 4.0 * NoV * NoL + 1e-4
    specular = numerator / denominator

    kd = 1.0 - fresnel
    if metallic is not None:
        kd *= (1.0 - metallic)

    render_rgb = (kd * albedo / np.pi + specular) * radiance * NoL
    render_rgb = torch.where(mask, render_rgb, background)

    if shadow is not None:
        render_rgb = torch.where(shadow == 0.0, render_rgb, render_rgb * 0.3)

    if linear:
        render_rgb = pbr_linear_to_srgb(render_rgb.squeeze())

    return {"render_rgb": render_rgb}


def getWorld2ViewTorch(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Rt = torch.zeros((4, 4), device=R.device, dtype=R.dtype)
    Rt[:3, :3] = R[:3, :3].T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


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

    def _compute_point_light_depth_cubemap(
        self, gaussians, position: torch.Tensor, resolution: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        device = position.device
        dtype = position.dtype
        res = max(16, int(resolution))

        _ = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)

        bg_color = torch.zeros((3, res, res), device=device, dtype=dtype)
        rotations = [
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                [
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            ),
        ]
        znear, zfar = 0.01, 100.0
        projection_matrix = (
            getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5, fovY=np.pi * 0.5)
            .transpose(0, 1)
            .to(device=device, dtype=dtype)
        )

        depth_faces: List[torch.Tensor] = []
        opacity_faces: List[torch.Tensor] = []
        for rotation in rotations:
            c2w = rotation.clone()
            c2w[:3, 3] = position
            w2c = torch.inverse(c2w)
            T = w2c[:3, 3]
            R = w2c[:3, :3].T
            world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
            full_proj_transform = (
                world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
            ).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            input_args = (
                bg_color,
                gaussians.get_xyz,
                torch.Tensor([]),
                gaussians.get_opacity,
                gaussians.get_scaling,
                gaussians.get_rotation,
                torch.Tensor([]),
                gaussians.get_features,
                camera_center,
                world_view_transform,
                full_proj_transform,
                1.0,
                1.0,
                1.0,
                res,
                res,
                gaussians.active_sh_degree,
                False,
                True,
            )
            try:
                (_, _, opacity_map, _, depth_map) = _C.lite_rasterize_gaussians(*input_args)
            except RuntimeError:
                return None
            depth_faces.append(depth_map.permute(1, 2, 0))
            opacity_faces.append(opacity_map.permute(1, 2, 0))

        return torch.stack(depth_faces), torch.stack(opacity_faces)

    def _get_point_light_depth_cubemap(
        self, gaussians, position: torch.Tensor, resolution: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        cache = self.state.get_point_light_shadow_cache()
        cached_pos = cache.get("position")
        cached_res = cache.get("resolution")
        depth_cubemap = cache.get("depth_cubemap")

        need_update = cache.get("dirty", True)
        if cached_res != int(resolution):
            need_update = True
        elif cached_pos is None or not torch.allclose(
            cached_pos.to(position.device), position, atol=1e-4, rtol=1e-3
        ):
            need_update = True
        elif depth_cubemap is None:
            need_update = True

        if need_update:
            result = self._compute_point_light_depth_cubemap(
                gaussians, position, int(resolution)
            )
            if result is None:
                cache["depth_cubemap"] = None
                cache["opacity_cubemap"] = None
                cache["dirty"] = True
            else:
                depth_cube, opacity_cube = result
                cache["depth_cubemap"] = depth_cube
                cache["opacity_cubemap"] = opacity_cube
                cache["dirty"] = False
            cache["position"] = position.detach().clone()
            cache["resolution"] = int(resolution)

        depth_cb = cache.get("depth_cubemap")
        opacity_cb = cache.get("opacity_cubemap")
        if depth_cb is None or opacity_cb is None:
            return None
        return depth_cb, opacity_cb

    def _compute_point_light_shading(
        self,
        gaussians,
        points: Optional[torch.Tensor],
        view_dirs: torch.Tensor,
        normal_map: torch.Tensor,
        normal_mask: torch.Tensor,
        albedo_map: torch.Tensor,
        roughness_map: torch.Tensor,
        metallic_map: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        point_state = self.state.point_light
        if not point_state.enabled or points is None:
            return None

        normals = normal_map.permute(1, 2, 0)
        mask = normal_mask.permute(1, 2, 0)
        albedo = albedo_map.permute(1, 2, 0)
        roughness = roughness_map.permute(1, 2, 0)
        metallic = metallic_map.permute(1, 2, 0) if self.state.enable_metallic else None
        position = self.state.get_point_light_position_tensor()
        intensity = self.state.get_point_light_intensity_tensor()

        shadow = None
        if point_state.enable_shadow:
            cubemap_pair = self._get_point_light_depth_cubemap(
                gaussians, position, point_state.shadow_resolution
            )
            if cubemap_pair is not None:
                depth_cubemap, _ = cubemap_pair
                depth_values = depth_cubemap.clone()
                max_depth = depth_values.max()
                depth_values[depth_values == 0] = max_depth
                to_light = position.view(1, 1, 3) - points
                distance_to_light = torch.norm(to_light, p=2, dim=-1, keepdim=True)
                query_dirs = F.normalize(-to_light, p=2, dim=-1)
                closest_depth = dr.texture(
                    depth_values[None, ...],
                    query_dirs[None, ...].contiguous(),
                    filter_mode="linear",
                    boundary_mode="cube",
                )[0]
                threshold = max(point_state.shadow_bias, 0.0)
                shadow = (distance_to_light - threshold > closest_depth).float()

        shading = light_pbr_shading(
            light_position=position,
            light_intensity=intensity,
            points=points,
            normals=normals,
            view_dirs=view_dirs,
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            mask=mask,
            shadow=shadow,
        )
        render_rgb = shading["render_rgb"]

        if self.state.enable_tone:
            render_rgb = pbr_aces_film(render_rgb)
        else:
            render_rgb = render_rgb.clamp(0.0, 1.0)
        if self.state.enable_gamma:
            render_rgb = pbr_linear_to_srgb(render_rgb)

        return render_rgb

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
                argmax_depth=True
            )
            t_after_render = time.perf_counter()

            normal_map = rendering_result["normal_map"]
            normal_mask = rendering_result["normal_mask"]
            opacity_mask = rendering_result["opacity_map"]
            albedo_map = rendering_result["albedo_map"]
            roughness_map = rendering_result["roughness_map"]
            metallic_map = rendering_result["metallic_map"]
            depth_map = rendering_result["depth_map"]

            # PBR mesh pass: render a simple pbr mesh

            H, W = view.image_height, view.image_width
            c2w = torch.inverse(view.world_view_transform.T)
            canonical_rays = get_canonical_rays(H, W, view.FoVx, view.FoVy)
            normalized_dirs = F.normalize(canonical_rays[:, None, :], p=2, dim=-1)
            view_dirs = -(
                (normalized_dirs * c2w[None, :3, :3]).sum(dim=-1).reshape(H, W, 3)
            )
            ray_norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(H, W, 1)

            points_world: Optional[torch.Tensor] = None
            if state.point_light.enabled:
                points_world = (
                    -view_dirs.reshape(-1, 3) * ray_norm.reshape(-1, 1) * depth_map.reshape(-1, 1)
                    + c2w[:3, 3]
                ).reshape(H, W, 3)

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
                mask=normal_mask.permute(1, 2, 0),
                albedo=albedo_map.permute(1, 2, 0),
                roughness=roughness_map.permute(1, 2, 0),
                metallic=metallic_map.permute(1, 2, 0) if state.enable_metallic else None,
                tone=state.enable_tone,
                gamma=state.enable_gamma,
                brdf_lut=state.brdf_lut,
            )

            render_rgb = result["render_rgb"].clamp(0.0, 1.0)

            point_light_rgb = self._compute_point_light_shading(
                gaussians=gaussians,
                points=points_world,
                view_dirs=view_dirs,
                normal_map=normal_map,
                normal_mask=normal_mask,
                albedo_map=albedo_map,
                roughness_map=roughness_map,
                metallic_map=metallic_map,
            )
            if point_light_rgb is not None:
                render_rgb = torch.clamp(render_rgb + point_light_rgb, 0.0, 1.0)
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
