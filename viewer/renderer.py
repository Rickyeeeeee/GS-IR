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
        self.mesh_context = self._create_mesh_context()
        self.meshes = list(self.state.loaded_meshes) if self.mesh_context is not None else []
        self._torch_profiler_enabled = bool(getattr(state.args, "torch_profiler", False))
        self._torch_profiler_ran = False
        self._torch_profiler_warning_emitted = False

    def _create_mesh_context(self):
        if self.state.device.type != "cuda":
            return None
        try:
            return dr.RasterizeCudaContext()
        except Exception:
            return None

    def _world_to_clip(self, positions: torch.Tensor, view) -> torch.Tensor:
        ones = torch.ones((positions.shape[0], 1), device=positions.device, dtype=positions.dtype)
        pos_h = torch.cat([positions, ones], dim=-1)
        view_matrix = view.world_view_transform.T
        proj_matrix = view.projection_matrix.T
        view_proj = proj_matrix @ view_matrix
        pos_clip = torch.matmul(pos_h, view_proj.T)
        return pos_clip[None, ...]

    def _render_mesh_gbuffer(
        self, view, mesh: Dict[str, torch.Tensor], height: int, width: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.mesh_context is None:
            return None

        pos_clip = self._world_to_clip(mesh["positions"], view)
        try:
            rast, _ = dr.rasterize(self.mesh_context, pos_clip, mesh["indices"], resolution=[height, width])
        except RuntimeError:
            return None

        mask = torch.clamp(rast[..., 3:], 0.0, 1.0)
        mask_bool = mask > 0.0

        normals, _ = dr.interpolate(mesh["normals"][None, ...], rast, mesh["indices"])
        normals = F.normalize(normals, dim=-1, eps=1e-6)
        world_pos, _ = dr.interpolate(mesh["positions"][None, ...], rast, mesh["indices"])

        uv_map = None
        if mesh.get("uvs") is not None:
            uv_interp, _ = dr.interpolate(mesh["uvs"][None, ...], rast, mesh["indices"])
            uv_map = uv_interp[..., :2]
            uv_map[..., 1] = 1.0 - uv_map[..., 1]

        device = normals.device
        H, W = mask.shape[1:3]

        base_factor = mesh["base_color_factor"].to(device).view(1, 1, 1, 3)
        albedo = base_factor.expand(1, H, W, 3).clone()
        base_texture = mesh.get("base_color_texture")
        if base_texture is not None and uv_map is not None:
            tex = base_texture
            tex_sample = dr.texture(
                tex[None, ...],
                uv_map.contiguous(),
                filter_mode="linear",
                boundary_mode="clamp",
            )
            albedo = albedo * tex_sample[..., :3]

        rough_factor = mesh["roughness_factor"].to(device).view(1, 1, 1, 1)
        metallic_factor = mesh["metallic_factor"].to(device).view(1, 1, 1, 1)
        roughness = rough_factor.expand(1, H, W, 1).clone()
        metallic = metallic_factor.expand(1, H, W, 1).clone()

        mr_texture = mesh.get("metallic_roughness_texture")
        if mr_texture is not None and uv_map is not None:
            tex = mr_texture
            mr_sample = dr.texture(
                tex[None, ...],
                uv_map.contiguous(),
                filter_mode="linear",
                boundary_mode="clamp",
            )
            if mr_sample.shape[-1] >= 3:
                metallic = metallic * mr_sample[..., 2:3].clamp(0.0, 1.0)
                roughness = roughness * mr_sample[..., 1:2].clamp(0.0, 1.0)
            elif mr_sample.shape[-1] == 2:
                roughness = roughness * mr_sample[..., 1:2].clamp(0.0, 1.0)
            elif mr_sample.shape[-1] >= 1:
                roughness = roughness * mr_sample[..., 0:1].clamp(0.0, 1.0)

        normals = torch.where(mask_bool.expand_as(normals), normals, torch.zeros_like(normals))
        world_pos = torch.where(mask_bool.expand_as(world_pos), world_pos, torch.zeros_like(world_pos))
        albedo = torch.where(mask_bool.expand_as(albedo), albedo, torch.zeros_like(albedo))
        roughness = torch.where(mask_bool.expand_as(roughness), roughness, torch.zeros_like(roughness))
        metallic = torch.where(mask_bool.expand_as(metallic), metallic, torch.zeros_like(metallic))

        return {
            "mask": mask[0].contiguous(),
            "mask_bool": mask_bool[0].contiguous(),
            "normal": normals[0].contiguous(),
            "albedo": albedo[0].contiguous(),
            "roughness": roughness[0].contiguous(),
            "metallic": metallic[0].contiguous(),
            "world": world_pos[0].contiguous(),
        }

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
        if not self._torch_profiler_enabled:
            return self._render_current_impl()

        if self._torch_profiler_ran:
            return self._render_current_impl()

        try:
            from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
        except ImportError:
            if not self._torch_profiler_warning_emitted:
                print("[viewer] torch.profiler is unavailable; skipping --torch_profiler.", flush=True)
                self._torch_profiler_warning_emitted = True
            self._torch_profiler_enabled = False
            return self._render_current_impl()

        activities = [ProfilerActivity.CPU]
        if self.state.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        sched = schedule(wait=1, warmup=2, active=5, repeat=1)  # adjust window to your step time

        with profile(
            activities=activities,
            # schedule=sched,
            on_trace_ready=tensorboard_trace_handler("logs/prof_run"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            result = self._render_current_impl()

        sort_key = "cuda_time_total" if self.state.device.type == "cuda" else "self_cpu_time_total"
        try:
            table = prof.key_averages().table(sort_by=sort_key, row_limit=50)
            print("[torch-profiler] render_current() results:\n" + table, flush=True)
        except ValueError:
            print("[torch-profiler] No events recorded during render_current().", flush=True)

        self._torch_profiler_ran = True
        return result

    def _render_current_impl(self) -> torch.Tensor:
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
                pad_normal=False,
                derive_normal=False,
                argmax_depth=False
            )
            t_after_render = time.perf_counter()

            normal_map = rendering_result["normal_map"]
            normal_mask = rendering_result["normal_mask"]
            opacity_mask = rendering_result["opacity_map"]
            albedo_map = rendering_result["albedo_map"]
            roughness_map = rendering_result["roughness_map"]
            metallic_map = rendering_result["metallic_map"]
            depth_map = rendering_result["depth_map"]

            H, W = view.image_height, view.image_width
            c2w = torch.inverse(view.world_view_transform.T)
            canonical_rays = get_canonical_rays(H, W, view.FoVx, view.FoVy)
            normalized_dirs = F.normalize(canonical_rays[:, None, :], p=2, dim=-1)
            view_dirs = -(
                (normalized_dirs * c2w[None, :3, :3]).sum(dim=-1).reshape(H, W, 3)
            )
            ray_norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(H, W, 1)

            camera_center = c2w[:3, 3]
            gaussian_points_world = (
                -view_dirs.reshape(-1, 3) * ray_norm.reshape(-1, 1) * depth_map.reshape(-1, 1)
                + camera_center
            ).reshape(H, W, 3)

            normal_hw = normal_map.permute(1, 2, 0)
            normal_mask_hw = normal_mask.permute(1, 2, 0)
            albedo_hw = albedo_map.permute(1, 2, 0)
            roughness_hw = roughness_map.permute(1, 2, 0)
            metallic_hw = metallic_map.permute(1, 2, 0)
            opacity_hw = opacity_mask.permute(1, 2, 0)

            combined_normal_hw = normal_hw
            combined_normal_mask_hw = normal_mask_hw
            combined_albedo_hw = albedo_hw
            combined_roughness_hw = roughness_hw
            combined_metallic_hw = metallic_hw
            combined_opacity_hw = opacity_hw
            combined_points_world = gaussian_points_world

            current_depth_world = torch.norm(
                gaussian_points_world - camera_center.view(1, 1, 3),
                dim=-1,
                keepdim=True,
            )
            current_depth_world = torch.where(
                opacity_hw > 0.0,
                current_depth_world,
                torch.full_like(current_depth_world, 1e6),
            )

            if self.mesh_context is not None and self.meshes:
                for mesh in self.meshes:
                    mesh_buffers = self._render_mesh_gbuffer(view, mesh, H, W)
                    if mesh_buffers is None:
                        continue

                    mesh_mask_hw = mesh_buffers["mask"]
                    mesh_mask_bool_hw = mesh_buffers["mask_bool"]
                    mesh_normal_hw = mesh_buffers["normal"]
                    mesh_albedo_hw = mesh_buffers["albedo"]
                    mesh_roughness_hw = mesh_buffers["roughness"]
                    mesh_metallic_hw = mesh_buffers["metallic"]
                    mesh_world_hw = mesh_buffers["world"]

                    mesh_depth_world = torch.norm(
                        mesh_world_hw - camera_center.view(1, 1, 3),
                        dim=-1,
                        keepdim=True,
                    )
                    mesh_depth_world = torch.where(
                        mesh_mask_bool_hw,
                        mesh_depth_world,
                        torch.full_like(mesh_depth_world, 1e6),
                    )

                    mesh_closer = torch.logical_and(mesh_depth_world < current_depth_world, mesh_mask_bool_hw)
                    mesh_closer_vec3 = mesh_closer.expand(-1, -1, 3)

                    combined_normal_hw = torch.where(mesh_closer_vec3, mesh_normal_hw, combined_normal_hw)
                    combined_albedo_hw = torch.where(mesh_closer_vec3, mesh_albedo_hw, combined_albedo_hw)
                    combined_points_world = torch.where(mesh_closer_vec3, mesh_world_hw, combined_points_world)
                    combined_metallic_hw = torch.where(mesh_closer, mesh_metallic_hw, combined_metallic_hw)
                    combined_roughness_hw = torch.where(mesh_closer, mesh_roughness_hw, combined_roughness_hw)
                    combined_opacity_hw = torch.where(mesh_closer, mesh_mask_hw, combined_opacity_hw)
                    combined_normal_mask_hw = torch.where(mesh_closer, mesh_mask_bool_hw, combined_normal_mask_hw)
                    current_depth_world = torch.where(mesh_closer, mesh_depth_world, current_depth_world)

            diff = camera_center.view(1, 1, 3) - combined_points_world
            depth_along_dir = torch.sum(diff * view_dirs, dim=-1, keepdim=True) / (ray_norm + 1e-6)
            depth_along_dir = torch.where(
                combined_opacity_hw > 0.0,
                depth_along_dir,
                torch.zeros_like(depth_along_dir),
            )
            depth_map = depth_along_dir.permute(2, 0, 1).contiguous()

            normal_map = combined_normal_hw.permute(2, 0, 1).contiguous()
            normal_mask = combined_normal_mask_hw.permute(2, 0, 1).contiguous()
            albedo_map = combined_albedo_hw.permute(2, 0, 1).contiguous()
            roughness_map = combined_roughness_hw.permute(2, 0, 1).contiguous()
            metallic_map = combined_metallic_hw.permute(2, 0, 1).contiguous()
            opacity_mask = combined_opacity_hw.permute(2, 0, 1).contiguous()

            t_after_mesh = time.perf_counter()

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

            points_world = combined_points_world if state.point_light.enabled else None
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
            "mesh": (t_after_mesh - t_after_render) * 1000.0,
            "shading": (t_after_shading - t_after_mesh) * 1000.0,
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

    def ensure_camera_matches_size(self, width: int, height: int) -> bool:
        return self.state.update_camera_resolution(width, height)

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
