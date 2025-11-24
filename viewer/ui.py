from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui, imguizmo

from .imgui_utils import imgui_cond, imgui_hovered_flag
from .state import ViewerState
from .renderer import ViewerRenderer
from utils.viewer_utils import euler_to_matrix


class ViewerUI:
    def __init__(self, state: ViewerState, renderer: ViewerRenderer, log_fn=print) -> None:
        self.state = state
        self.renderer = renderer

        self.camera_first_update = False

        self.log = log_fn

        self.render_window_hovered = False
        self.render_window_focused = False
        self.render_image_hovered = False
        self.render_image_active = False
        self._gizmo = imguizmo.im_guizmo
        self._gizmo_identity = np.eye(4, dtype=np.float32)
        self._gizmo_operation = self._gizmo.OPERATION.translate
        self._gizmo_mode = self._gizmo.MODE.local
        base_rotation = (
            self.state.base_R_fix[:3, :3].detach().cpu().numpy().astype(np.float32)
            if hasattr(self.state, "base_R_fix")
            else np.eye(3, dtype=np.float32)
        )
        self._gizmo_base_rotation = base_rotation
        self._gizmo_base_rotation_inv = base_rotation.T
        self._gizmo_target_type = "model"  # "model" or "mesh"
        self._gizmo_target_mesh_idx = 0

    # --------------------------------------------------------------------- windows --
    def draw_control_window(self, last_frame_dt: float, viewer_timings: dict[str, float]) -> None:
        cond_first = imgui_cond("FirstUseEver")
        if cond_first is not None:
            imgui.set_next_window_size((360, self.state.args.height - 40), cond_first)
            imgui.set_next_window_pos((20, 10), cond_first)

        if not imgui.begin("Controls"):
            imgui.end()
            return

        fps = 1.0 / last_frame_dt if last_frame_dt > 0 else 0.0
        imgui.text(f"FPS: {fps:.1f}")
        imgui.text(f"Frame Time: {last_frame_dt * 1000.0:.2f} ms")
        imgui.separator()

        indent = getattr(imgui, "indent", None)
        unindent = getattr(imgui, "unindent", None)

        if imgui.collapsing_header("Rendering", imgui.TreeNodeFlags_.default_open):
            if callable(indent):
                indent()
            render_mode_index = 1 if self.state.render_jointly else 0
            render_modes = ["Active Model", "Joint (Concatenated)"]
            changed, new_mode_idx = imgui.combo("Render Mode", render_mode_index, render_modes)
            if changed:
                self.state.render_jointly = new_mode_idx == 1
                self.state.mark_all_models_dirty()
                self.log(f"Render mode set to {render_modes[new_mode_idx]}")

            if imgui.begin_combo("Active Model", self.state.model_names[self.state.active_model_idx]):
                for idx, name in enumerate(self.state.model_names):
                    selected = idx == self.state.active_model_idx
                    if imgui.selectable(name, selected)[0] and not selected:
                        self.state.active_model_idx = idx
                        self.state.mark_model_dirty(idx)
                        self.log(f"Active model set to {name}")
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()

            imgui.separator()
            imgui.text("Render Resolution")
            scale = self.renderer.resolution_scale
            changed, new_scale = imgui.slider_float("Resolution Scale", scale, 0.25, 1.0, format="%.2f")
            if changed:
                self.renderer.set_resolution_scale(new_scale)
                self.log(f"Resolution scale set to {new_scale:.2f}")
            imgui.text(
                f"Render Size: {int(self.state.camera.image_width)} x {int(self.state.camera.image_height)}"
            )
            if callable(unindent):
                unindent()

        if imgui.collapsing_header("Transforms & Gizmo", imgui.TreeNodeFlags_.default_open):
            if callable(indent):
                indent()
            self._draw_transform_controls()
            if callable(unindent):
                unindent()

        if imgui.collapsing_header("Environment & Lighting", imgui.TreeNodeFlags_.default_open):
            if callable(indent):
                indent()
            self._draw_environment_controls()
            if callable(unindent):
                unindent()

        if imgui.collapsing_header("Camera", imgui.TreeNodeFlags_.default_open):
            if callable(indent):
                indent()
            current_fovx = self.state.get_camera_fovx_degrees()
            changed_fov, new_fovx = imgui.slider_float("FoVx (deg)", current_fovx, 10.0, 170.0, "%.1f")
            if changed_fov:
                if self.state.set_camera_fovx_degrees(new_fovx):
                    self.log(f"Camera FoVx set to {new_fovx:.1f} deg")
            if callable(unindent):
                unindent()

        if imgui.collapsing_header("Profiling & Stats", imgui.TreeNodeFlags_.default_open):
            if callable(indent):
                indent()
            imgui.text("Profiling")
            profiler_available = self.renderer.is_profiler_available()
            begin_disabled = getattr(imgui, "begin_disabled", None)
            end_disabled = getattr(imgui, "end_disabled", None)
            disabled_scope = not profiler_available and callable(begin_disabled) and callable(end_disabled)
            if disabled_scope:
                begin_disabled(True)
            if imgui.button("Profile Next Frame"):
                if self.renderer.request_profiler_capture():
                    self.log("Queued torch.profiler capture for next frame")
                else:
                    self.log("Torch profiler unavailable; capture request ignored")
            if disabled_scope:
                end_disabled()
                imgui.same_line()
                imgui.text_disabled("torch.profiler unavailable")
            elif not profiler_available:
                imgui.text_disabled("torch.profiler unavailable")

            timings = self.state.profile_timings
            if timings:
                imgui.separator()
                imgui.text("Render Timings")
                imgui.text(f"Prepare: {timings.get('prepare', float('nan')):.2f} ms")
                imgui.text(f"GS Render: {timings.get('render', float('nan')):.2f} ms")
                imgui.text(f"Mesh Render: {timings.get('mesh', float('nan')):.2f} ms")
                imgui.text(f"Shading: {timings.get('shading', float('nan')):.2f} ms")
                imgui.text(f"Composite: {timings.get('composite', float('nan')):.2f} ms")
                imgui.text(f"Total: {timings.get('total', float('nan')):.2f} ms")
                if "gaussian_count" in timings:
                    imgui.text(f"Gaussians: {int(timings['gaussian_count'])}")
            else:
                imgui.text("No timing data yet.")

            if viewer_timings:
                imgui.separator()
                imgui.text("Viewer Timing")

                def _fmt(key: str) -> str:
                    value = viewer_timings.get(key)
                    if value is None or not np.isfinite(value):
                        return "--"
                    return f"{value:.2f} ms"

                imgui.text(f"Input: {_fmt('input')}")
                imgui.text(f"Update Render: {_fmt('update_render')}")
                imgui.text(f"Draw Controls: {_fmt('draw_control')}")
                imgui.text(f"Draw Render: {_fmt('draw_render')}")
                imgui.text(f"Image Display: {_fmt('image_display')}")
                imgui.text(f"Frame (UI): {_fmt('frame')}")
            if callable(unindent):
                unindent()

        if imgui.collapsing_header("Actions", imgui.TreeNodeFlags_.default_open):
            if callable(indent):
                indent()
            if imgui.button("Render Once"):
                self.renderer.update_render_buffer()
                self.log("Render Once button pressed")
            imgui.same_line()
            if imgui.button("Save Transforms"):
                self.state.save_transform_state()
                self.log("Save Transforms button pressed")

            if self.state.transform_state_path:
                imgui.text_wrapped(f"Transforms file: {self.state.transform_state_path}")

            imgui.separator()
            imgui.text("Checkpoints:")
            for ckpt in self.state.args.checkpoint:
                imgui.text_wrapped(ckpt)
            if callable(unindent):
                unindent()

        imgui.end()

    def draw_render_window(self) -> None:
        cond_first = imgui_cond("FirstUseEver")
        if cond_first is not None:
            imgui.set_next_window_size((self.state.args.width - 400, self.state.args.height - 40), cond_first)
            imgui.set_next_window_pos((400, 10), cond_first)

        if imgui.begin("Render"):
            if self.renderer.last_render_error:
                error_color = imgui.ImVec4(1.0, 0.2, 0.2, 1.0)
                imgui.text_colored(error_color, f"Render error: {self.renderer.last_render_error}")
                self.render_image_hovered = False
                self.render_image_active = False
            elif self.renderer.has_image:
                get_region = getattr(imgui, "get_content_region_avail", getattr(imgui, "get_content_region_available", None))
                avail = get_region() if get_region else (self.state.camera.image_width, self.state.camera.image_height)
                raw_w = int(avail.x if hasattr(avail, "x") else avail[0])
                raw_h = int(avail.y if hasattr(avail, "y") else avail[1])
                avail_w = max(1, raw_w)
                avail_h = max(1, raw_h)

                scale = self.renderer.resolution_scale
                target_w = max(1, int(avail_w * scale))
                target_h = max(1, int(avail_h * scale))
                changed = self.renderer.ensure_camera_matches_size(target_w, target_h)
                if not self.camera_first_update and changed:
                    self.camera_first_update = True
                    self.renderer.resolution_scale = 720 / self.state.camera.image_height
                    

                texture_id, upload_ms, texture_error = self.renderer.ensure_render_texture()
                self.renderer.image_display_time_ms = upload_ms
                if texture_error:
                    imgui.text_colored(f"Cannot display image: {texture_error}", 1.0, 0.2, 0.2, 1.0)
                    self.render_image_hovered = False
                    self.render_image_active = False
                elif texture_id is not None:
                    render_w = max(1, int(self.state.camera.image_width))
                    render_h = max(1, int(self.state.camera.image_height))
                    display_w = float(avail_w)
                    display_h = float(avail_w * render_h / render_w) if render_w > 0 else float(avail_h)
                    if display_h > avail_h and render_h > 0:
                        display_h = float(avail_h)
                        display_w = float(avail_h * render_w / render_h)

                    size_vec = imgui.ImVec2(display_w, display_h)
                    imgui.image(texture_id, size_vec)
                    if display_h < avail_h:
                        imgui.dummy((0.0, float(avail_h - display_h)))
                    self.render_image_hovered = bool(imgui.is_item_hovered())
                    self.render_image_active = bool(imgui.is_item_active())
                    self._draw_gizmo_widget()
                else:
                    imgui.text("Texture unavailable; ensure an OpenGL context is active.")
                    self.render_image_hovered = False
                    self.render_image_active = False
            else:
                imgui.text("Rendering...")
                self.render_image_hovered = False
                self.render_image_active = False

            hovered_flag_none = imgui_hovered_flag("none", 0)
            self.render_window_hovered = bool(imgui.is_window_hovered(hovered_flag_none))
            self.render_window_focused = bool(imgui.is_window_focused())
        else:
            self.render_window_hovered = False
            self.render_window_focused = False
            self.render_image_hovered = False
            self.render_image_active = False
        imgui.end()

    # ----------------------------------------------------------------- sub-widgets --
    def _draw_gizmo_widget(self) -> None:
        gizmo = self._gizmo
        gizmo.begin_frame()
        gizmo.set_orthographic(False)
        window_pos = imgui.get_window_pos()
        content_min = imgui.get_window_content_region_min()
        content_max = imgui.get_window_content_region_max()
        rect_x = window_pos.x + content_min.x
        rect_y = window_pos.y + content_min.y
        rect_w = content_max.x - content_min.x
        rect_h = content_max.y - content_min.y
        gizmo.set_drawlist()
        gizmo.set_rect(rect_x, rect_y, rect_w, rect_h)

        camera = self.state.camera
        corrected_world_view_transform = camera.world_view_transform.clone()
        corrected_world_view_transform[:, 1] *= -1
        camera_view = np.ascontiguousarray(corrected_world_view_transform.cpu().numpy(), dtype=np.float32)
        camera_projection = np.ascontiguousarray(camera.projection_matrix.cpu().numpy(), dtype=np.float32)
        object_matrix = self._build_gizmo_matrix()

        gizmo.draw_grid(camera_view, camera_projection, self._gizmo_identity, 10.0)
        manip_result = gizmo.manipulate(
            camera_view,
            camera_projection,
            self._gizmo_operation,
            self._gizmo_mode,
            object_matrix,
            None,
            None,
            None,
            None,
        )
        if manip_result:
            result_matrix = np.asarray(manip_result.value, dtype=np.float32)
            self._apply_gizmo_transform(np.ascontiguousarray(result_matrix.T))

    def _build_gizmo_matrix(self) -> np.ndarray:
        trans, target_type, _ = self._get_gizmo_target_transform()
        if trans is None:
            return np.eye(4, dtype=np.float32)

        yaw = math.radians(float(trans.get("yaw", 0.0)))
        pitch = math.radians(float(trans.get("pitch", 0.0)))
        roll = math.radians(float(trans.get("roll", 0.0)))
        rotation_user = euler_to_matrix(yaw, pitch, roll).astype(np.float32)
        base_rot = self._gizmo_base_rotation if target_type == "model" else np.eye(3, dtype=np.float32)
        rotation_combined = rotation_user @ base_rot
        scale = float(trans.get("scale", 1.0))
        translation = np.array(trans.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation_combined * scale
        matrix[:3, 3] = translation
        return np.ascontiguousarray(matrix.T)

    def _apply_gizmo_transform(self, matrix: np.ndarray) -> None:
        trans, target_type, target_idx = self._get_gizmo_target_transform()
        if trans is None:
            return

        scale = float(np.linalg.norm(matrix[:3, 0]))
        scale = max(scale, 1e-6)
        rotation_combined = matrix[:3, :3] / scale
        base_rot = self._gizmo_base_rotation if target_type == "model" else np.eye(3, dtype=np.float32)
        base_rot_inv = self._gizmo_base_rotation_inv if target_type == "model" else np.eye(3, dtype=np.float32)
        rotation_user = rotation_combined @ base_rot_inv
        yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_user)
        translation = matrix[:3, 3]

        trans["translation"] = translation.astype(np.float32).tolist()
        trans["scale"] = scale
        trans["yaw"] = math.degrees(yaw)
        trans["pitch"] = math.degrees(pitch)
        trans["roll"] = math.degrees(roll)
        if target_type == "model":
            self.state.mark_model_dirty(target_idx)
        else:
            # Mesh data is transformed at render time; no cache to invalidate.
            pass

    def _rotation_matrix_to_euler(self, rotation: np.ndarray) -> tuple[float, float, float]:
        trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rotation[2, 1] - rotation[1, 2]) / s
            qy = (rotation[0, 2] - rotation[2, 0]) / s
            qz = (rotation[1, 0] - rotation[0, 1]) / s
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll_x = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch_y = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch_y = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_z = math.atan2(siny_cosp, cosy_cosp)

        pitch = roll_x
        yaw = pitch_y
        roll = yaw_z
        return yaw, pitch, roll

    def _get_gizmo_target_transform(self) -> tuple[dict | None, str, int]:
        if self._gizmo_target_type == "mesh":
            groups = getattr(self.state, "mesh_group_transforms", [])
            if not groups:
                return None, "mesh", -1
            idx = min(max(int(self._gizmo_target_mesh_idx), 0), len(groups) - 1)
            self._gizmo_target_mesh_idx = idx
            return groups[idx], "mesh", idx
        return self.state.model_transforms[self.state.active_model_idx], "model", self.state.active_model_idx

    def _draw_transform_controls(self) -> None:
        trans, target_type, target_idx = self._get_gizmo_target_transform()
        if trans is None:
            imgui.text_disabled("No gizmo target available.")
            return

        # Target selector
        options = [f"Model: {name}" for name in self.state.model_names]
        mesh_labels = getattr(self.state, "mesh_group_names", [])
        options.extend([f"Mesh: {label}" for label in mesh_labels])
        current_idx = target_idx if target_type == "model" else len(self.state.model_names) + target_idx
        if options:
            changed, new_idx = imgui.combo("Gizmo Target", current_idx, options)
            if changed:
                if new_idx < len(self.state.model_names):
                    self._gizmo_target_type = "model"
                    self.state.active_model_idx = new_idx
                else:
                    self._gizmo_target_type = "mesh"
                    self._gizmo_target_mesh_idx = new_idx - len(self.state.model_names)
                trans, target_type, target_idx = self._get_gizmo_target_transform()

        def slider(label: str, field: str, min_v: float, max_v: float):
            changed, value = imgui.slider_float(label, trans[field], min_v, max_v)
            if changed:
                trans[field] = value
                if target_type == "model":
                    self.state.mark_model_dirty(self.state.active_model_idx)
                self.log(f"{label} set to {value:.3f}")

        if imgui.tree_node("Gizmo"):
            if callable(imgui.indent):
                imgui.indent()
            imgui.text("Gizmo Operation")
            if imgui.radio_button("Translate", self._gizmo_operation == self._gizmo.OPERATION.translate):
                self._gizmo_operation = self._gizmo.OPERATION.translate
            imgui.same_line()
            if imgui.radio_button("Rotate", self._gizmo_operation == self._gizmo.OPERATION.rotate):
                self._gizmo_operation = self._gizmo.OPERATION.rotate
            imgui.same_line()
            if imgui.radio_button("Scale", self._gizmo_operation == self._gizmo.OPERATION.scale):
                self._gizmo_operation = self._gizmo.OPERATION.scale
            if callable(imgui.unindent):
                imgui.unindent()
            imgui.tree_pop()

        if imgui.tree_node("Model"):
            if callable(imgui.indent):
                imgui.indent()
            imgui.text("Orientation")
            slider("Yaw (deg)", "yaw", -180.0, 180.0)
            slider("Pitch (deg)", "pitch", -180.0, 180.0)
            slider("Roll (deg)", "roll", -180.0, 180.0)

            imgui.separator()
            imgui.text("Translation")
            changed, translation = imgui.drag_float3("XYZ", trans["translation"], v_speed=0.01, format="%.3f")
            if changed:
                trans["translation"] = [float(v) for v in translation]
                if target_type == "model":
                    self.state.mark_model_dirty(self.state.active_model_idx)
                self.log("Translation updated")

            changed, scale_value = imgui.slider_float("Uniform Scale", trans["scale"], 0.0001, 10.0, format="%.3f")
            if changed:
                trans["scale"] = max(scale_value, 1e-6)
                if target_type == "model":
                    self.state.mark_model_dirty(self.state.active_model_idx)
                self.log(f"Uniform Scale set to {trans['scale']:.3f}")
            if callable(imgui.unindent):
                imgui.unindent()
            imgui.tree_pop()

        if target_type == "model" and imgui.tree_node("Bounding Box"):
            if callable(imgui.indent):
                imgui.indent()
            changed_min, bbox_min = imgui.drag_float3("BBox Min", trans["bbox_min"], v_speed=0.01, format="%.3f")
            if changed_min:
                trans["bbox_min"] = [float(v) for v in bbox_min]
                self.state.ensure_bbox_consistency(trans)
                self.state.mark_model_dirty(self.state.active_model_idx)
                self.log("BBox Min updated")

            changed_max, bbox_max = imgui.drag_float3("BBox Max", trans["bbox_max"], v_speed=0.01, format="%.3f")
            if changed_max:
                trans["bbox_max"] = [float(v) for v in bbox_max]
                self.state.ensure_bbox_consistency(trans)
                self.state.mark_model_dirty(self.state.active_model_idx)
                self.log("BBox Max updated")

            for label, key in (("BBox Yaw", "bbox_yaw"), ("BBox Pitch", "bbox_pitch"), ("BBox Roll", "bbox_roll")):
                current = float(trans.get(key, 0.0))
                changed_angle, value_angle = imgui.slider_float(label, current, -180.0, 180.0)
                if changed_angle:
                    trans[key] = float(value_angle)
                    self.state.mark_model_dirty(self.state.active_model_idx)
                    self.log(f"{label} updated to {value_angle:.2f}")
            if callable(imgui.unindent):
                imgui.unindent()
            imgui.tree_pop()

    def _draw_environment_controls(self) -> None:
        if imgui.tree_node("Environment Map"):
            if callable(imgui.indent):
                imgui.indent()
            if imgui.begin_combo("HDRI Preset", self.state.hdri_label_current):
                for label in self.state.hdri_labels:
                    selected = label == self.state.hdri_label_current
                    if imgui.selectable(label, selected)[0] and not selected:
                        self.state.hdri_label_current = label
                        self.state.ensure_hdri(label)
                        self.state.mark_all_models_dirty()
                        self.log(f"HDRI preset set to {label}")
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()

            changed, rotation = imgui.slider_float("HDRI Yaw", self.state.hdri_rotation_deg, -180.0, 180.0, "%.1f deg")
            if changed:
                self.state.hdri_rotation_deg = rotation
                self.log(f"HDRI yaw set to {rotation:.1f} degrees")
            if callable(imgui.unindent):
                imgui.unindent()
            imgui.tree_pop()

        if imgui.tree_node("Shading"):
            if callable(imgui.indent):
                imgui.indent()
            changed, tone_state = imgui.checkbox("ACES tone mapping", self.state.enable_tone)
            if changed:
                self.state.enable_tone = tone_state
                self.log(f"ACES tone mapping {'enabled' if tone_state else 'disabled'}")

            changed, gamma_state = imgui.checkbox("Gamma correction (sRGB)", self.state.enable_gamma)
            if changed:
                self.state.enable_gamma = gamma_state
                self.log(f"Gamma correction {'enabled' if gamma_state else 'disabled'}")

            changed, env_bg_state = imgui.checkbox("HDRI as background", self.state.show_env_bg)
            if changed:
                self.state.show_env_bg = env_bg_state
                self.log(f"HDRI background {'enabled' if env_bg_state else 'disabled'}")
            if callable(imgui.unindent):
                imgui.unindent()
            imgui.tree_pop()

        if imgui.tree_node("Point Light"):
            if callable(imgui.indent):
                imgui.indent()
            point = self.state.point_light
            changed_light, enabled = imgui.checkbox("Enable Point Light", point.enabled)
            if changed_light:
                point.enabled = enabled
                self.log(f"Point light {'enabled' if enabled else 'disabled'}")

            if point.enabled:
                changed_pos, new_pos = imgui.drag_float3(
                    "Position", point.position, v_speed=0.01, format="%.3f"
                )
                if changed_pos:
                    point.position = [float(v) for v in new_pos]
                    self.state.mark_point_light_dirty()
                    self.log(f"Point light position set to {point.position}")

                changed_intensity, new_intensity = imgui.drag_float3(
                    "Intensity", point.intensity, v_speed=1.0, format="%.2f"
                )
                if changed_intensity:
                    point.intensity = [max(0.0, float(v)) for v in new_intensity]
                    self.log("Point light intensity updated")

                changed_shadow, shadow_state = imgui.checkbox("Cast Shadows", point.enable_shadow)
                if changed_shadow:
                    point.enable_shadow = shadow_state
                    self.state.mark_point_light_dirty()
                    self.log(f"Point light shadows {'enabled' if shadow_state else 'disabled'}")

                if point.enable_shadow:
                    shadow_res_options = [128, 256, 512, 1024, 2048]
                    if point.shadow_resolution not in shadow_res_options:
                        shadow_res_options.append(point.shadow_resolution)
                        shadow_res_options.sort()
                    current_idx = shadow_res_options.index(point.shadow_resolution)
                    labels = [f"{res}" for res in shadow_res_options]
                    changed_res, new_idx = imgui.combo("Shadow Resolution", current_idx, labels)
                    if changed_res and 0 <= new_idx < len(shadow_res_options):
                        point.shadow_resolution = int(shadow_res_options[new_idx])
                        self.state.mark_point_light_dirty()
                        self.log(f"Point light shadow resolution set to {point.shadow_resolution}")

                    changed_bias, new_bias = imgui.drag_float(
                        "Shadow Threshold", point.shadow_bias, v_speed=0.001, v_min=0.0, v_max=1.0, format="%.4f"
                    )
                    if changed_bias:
                        point.shadow_bias = float(max(0.0, min(1.0, new_bias)))
                        self.log(f"Point light shadow threshold set to {point.shadow_bias:.4f}")
            if callable(imgui.unindent):
                imgui.unindent()
            imgui.tree_pop()
