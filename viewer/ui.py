from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, imguizmo

from .imgui_utils import imgui_cond, imgui_hovered_flag
from .state import ViewerState
from .renderer import ViewerRenderer


class ViewerUI:
    def __init__(self, state: ViewerState, renderer: ViewerRenderer, log_fn=print) -> None:
        self.state = state
        self.renderer = renderer
        self.log = log_fn

        self.render_window_hovered = False
        self.render_window_focused = False
        self.render_image_hovered = False
        self.render_image_active = False
        self._gizmo = imguizmo.im_guizmo
        self._gizmo_object_matrix = np.array(
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            dtype=np.float32,
        )
        self._gizmo_identity = np.eye(4, dtype=np.float32)
        self._gizmo_operation = self._gizmo.OPERATION.translate
        self._gizmo_mode = self._gizmo.MODE.local

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

        render_mode_index = 1 if self.state.render_jointly else 0
        render_modes = ["Active Model", "Joint (Concatenated)"]
        changed, new_mode_idx = imgui.combo("Render Mode", render_mode_index, render_modes)
        if changed:
            self.state.render_jointly = new_mode_idx == 1
            self.state.mark_all_models_dirty()
            self.log(f"Render mode set to {render_modes[new_mode_idx]}")

        imgui.separator()
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
        self._draw_transform_controls()
        imgui.separator()
        self._draw_environment_controls()

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

        imgui.separator()
        imgui.text("Profiling")
        timings = self.state.profile_timings
        if timings:
            imgui.text(f"Prepare: {timings.get('prepare', float('nan')):.2f} ms")
            imgui.text(f"Render: {timings.get('render', float('nan')):.2f} ms")
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

        imgui.separator()
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
                self.renderer.ensure_camera_matches_size(target_w, target_h)

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
                    imgui.text(
                        f"Displayed {render_w} x {render_h} (scale {self.renderer.resolution_scale:.2f})"
                    )
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
        corrected_world_view_transform[:,1] *= -1
        camera_view = np.ascontiguousarray(corrected_world_view_transform.cpu().numpy(), dtype=np.float32)
        camera_projection = np.ascontiguousarray(camera.projection_matrix.cpu().numpy(), dtype=np.float32)
        object_matrix = np.ascontiguousarray(self._gizmo_object_matrix, dtype=np.float32)

        gizmo.draw_grid(camera_view, camera_projection, self._gizmo_identity, 10.0)
        gizmo.draw_cubes(camera_view, camera_projection, [object_matrix])

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
            self._gizmo_object_matrix = np.ascontiguousarray(manip_result.value.astype(np.float32))

    def _draw_transform_controls(self) -> None:
        trans = self.state.model_transforms[self.state.active_model_idx]

        def slider(label: str, field: str, min_v: float, max_v: float):
            changed, value = imgui.slider_float(label, trans[field], min_v, max_v)
            if changed:
                trans[field] = value
                self.state.mark_model_dirty(self.state.active_model_idx)
                self.log(f"{label} set to {value:.3f}")

        imgui.text("Model Orientation")
        slider("Yaw (deg)", "yaw", -180.0, 180.0)
        slider("Pitch (deg)", "pitch", -180.0, 180.0)
        slider("Roll (deg)", "roll", -180.0, 180.0)

        imgui.separator()
        imgui.text("Translation")
        changed, translation = imgui.drag_float3("XYZ", trans["translation"], v_speed=0.01, format="%.3f")
        if changed:
            trans["translation"] = [float(v) for v in translation]
            self.state.mark_model_dirty(self.state.active_model_idx)
            self.log("Translation updated")

        imgui.separator()
        changed, scale_value = imgui.slider_float("Uniform Scale", trans["scale"], 0.0001, 10.0, format="%.3f")
        if changed:
            trans["scale"] = max(scale_value, 1e-6)
            self.state.mark_model_dirty(self.state.active_model_idx)
            self.log(f"Uniform Scale set to {trans['scale']:.3f}")

        imgui.separator()
        imgui.text("Bounding Box")
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

    def _draw_environment_controls(self) -> None:
        imgui.text("Environment")
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

        imgui.separator()
        imgui.text("Shading")
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
