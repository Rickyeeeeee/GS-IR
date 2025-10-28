from __future__ import annotations

import time

from imgui_bundle import immapp, hello_imgui, imgui

from .camera_controller import CameraController
from .renderer import ViewerRenderer
from .state import ViewerState
from .ui import ViewerUI


class RelightViewer:
    def __init__(self, args):
        self.args = args
        self.state = ViewerState(args)
        self.renderer = ViewerRenderer(self.state)
        self.controller = CameraController(self.state.camera, log_fn=self._log_input)
        self.ui = ViewerUI(self.state, self.renderer, log_fn=self._log_input)

        self.viewer_timings: dict[str, float] = {}
        self.total_frames = 0
        self._last_frame_time = time.perf_counter()

    def _log_input(self, message: str) -> None:
        print(f"[input] {message}", flush=True)

    def gui_frame(self):
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now

        t_start = now
        self.controller.process_inputs(dt)
        t_after_input = time.perf_counter()

        self.renderer.update_render_buffer()
        t_after_update = time.perf_counter()

        self.ui.draw_control_window(self.controller.last_dt, self.viewer_timings)
        t_after_control = time.perf_counter()

        self.ui.draw_render_window()
        t_after_render = time.perf_counter()

        self.controller.update_window_state(
            self.ui.render_window_hovered,
            self.ui.render_window_focused,
            self.ui.render_image_hovered,
            self.ui.render_image_active,
        )

        self.viewer_timings = {
            "input": (t_after_input - t_start) * 1000.0,
            "update_render": (t_after_update - t_after_input) * 1000.0,
            "draw_control": (t_after_control - t_after_update) * 1000.0,
            "draw_render": (t_after_render - t_after_control) * 1000.0,
            "image_display": self.renderer.image_display_time_ms,
            "frame": (t_after_render - t_start) * 1000.0,
        }

        self.total_frames += 1

    def run(self):
        runner_params = hello_imgui.RunnerParams()
        runner_params.callbacks.show_gui = self.gui_frame

        runner_params.app_window_params.window_title = "GS-IR PBR Viewer"
        runner_params.app_window_params.window_geometry.size = (self.args.width, self.args.height)
        runner_params.app_window_params.restore_previous_geometry = True

        runner_params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
        )

        runner_params.fps_idling.enable_idling = False
        runner_params.fps_idling.fps_idle = 0.0

        default_setup = runner_params.callbacks.setup_imgui_config

        def setup_imgui_config() -> None:
            if callable(default_setup):
                default_setup()
            io = imgui.get_io()
            docking_flag = None
            config_enum = getattr(imgui, "ConfigFlags_", None)
            if config_enum is not None:
                docking_flag = getattr(config_enum, "docking_enable", None)
            if docking_flag is None:
                config_class = getattr(imgui, "ConfigFlags", None)
                docking_flag = getattr(config_class, "DockingEnable", None) if config_class else None
            if docking_flag is None:
                docking_flag = 1 << 6  # ImGuiConfigFlags_DockingEnable fallback
            io.config_flags |= int(docking_flag)

        runner_params.callbacks.setup_imgui_config = setup_imgui_config

        immapp.run(runner_params)

def run_viewer(args) -> None:
    RelightViewer(args).run()
