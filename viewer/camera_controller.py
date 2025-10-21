from __future__ import annotations

from typing import Callable

from imgui_bundle import imgui

from .imgui_utils import imgui_key, imgui_mouse_button


class CameraController:
    def __init__(
        self,
        camera,
        move_speed: float = 4.0,
        orbit_key_speed: float = 4.0,
        mouse_sensitivity: float = 0.002,
        max_dt: float = 0.10,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.camera = camera
        self.move_speed = move_speed
        self.orbit_key_speed = orbit_key_speed
        self.mouse_sensitivity = mouse_sensitivity
        self.max_dt = max_dt
        self.log = log_fn or (lambda msg: None)

        self._key_cache = {
            "forward": imgui_key("W"),
            "back": imgui_key("S"),
            "left": imgui_key("A"),
            "right": imgui_key("D"),
            "up": imgui_key("Q"),
            "down": imgui_key("E"),
            "yaw_left": imgui_key("LeftArrow"),
            "yaw_right": imgui_key("RightArrow"),
            "pitch_up": imgui_key("UpArrow"),
            "pitch_down": imgui_key("DownArrow"),
        }
        self._key_names = {
            "forward": "W",
            "back": "S",
            "left": "A",
            "right": "D",
            "up": "Q",
            "down": "E",
            "yaw_left": "LeftArrow",
            "yaw_right": "RightArrow",
            "pitch_up": "UpArrow",
            "pitch_down": "DownArrow",
        }
        self._key_state = {name: False for name in self._key_cache}

        mouse_button = imgui_mouse_button("Left")
        self._mouse_button = int(mouse_button) if mouse_button is not None else 0
        self._mouse_dragging = False

        self.render_window_hovered = False
        self.render_window_focused = False
        self.render_image_hovered = False
        self.render_image_active = False

        self.last_dt = 0.0

    def update_window_state(
        self,
        window_hovered: bool,
        window_focused: bool,
        image_hovered: bool,
        image_active: bool,
    ) -> None:
        self.render_window_hovered = window_hovered
        self.render_window_focused = window_focused
        self.render_image_hovered = image_hovered
        self.render_image_active = image_active

    def _is_down(self, key_name: str) -> bool:
        key = self._key_cache.get(key_name)
        if key is None:
            return False
        return bool(imgui.is_key_down(key))

    def _update_key_log(self, logical_name: str, is_down: bool) -> None:
        prev = self._key_state.get(logical_name, False)
        if is_down != prev:
            self._key_state[logical_name] = is_down
            label = self._key_names.get(logical_name, logical_name)
            state = "down" if is_down else "up"
            self.log(f"key {label} {state}")

    def process_inputs(self, dt: float) -> None:
        io = imgui.get_io()
        dt = max(0.0, min(dt, self.max_dt))
        if dt <= 0.0:
            return False

        self.last_dt = dt

        allow_keyboard = not io.want_capture_keyboard or self.render_window_focused or self.render_window_hovered
        allow_mouse = (
            not io.want_capture_mouse
            or self.render_window_hovered
            or self.render_image_hovered
            or self.render_image_active
        )


        if allow_keyboard:
            key_states = {
                "forward": self._is_down("forward"),
                "back": self._is_down("back"),
                "up": self._is_down("up"),
                "down": self._is_down("down"),
                "right": self._is_down("right"),
                "left": self._is_down("left"),
                "yaw_left": self._is_down("yaw_left"),
                "yaw_right": self._is_down("yaw_right"),
                "pitch_up": self._is_down("pitch_up"),
                "pitch_down": self._is_down("pitch_down"),
            }

            for name, state in key_states.items():
                self._update_key_log(name, state)

            if key_states["forward"]:
                self.camera.move_forward(+self.move_speed * dt)
            if key_states["back"]:
                self.camera.move_forward(-self.move_speed * dt)
            if key_states["up"]:
                self.camera.move_up(+self.move_speed * dt)
            if key_states["down"]:
                self.camera.move_up(-self.move_speed * dt)
            if key_states["right"]:
                self.camera.move_right(+self.move_speed * dt)
            if key_states["left"]:
                self.camera.move_right(-self.move_speed * dt)

            yaw_delta = 0.0
            pitch_delta = 0.0
            if key_states["yaw_left"]:
                yaw_delta += +self.orbit_key_speed * dt
            if key_states["yaw_right"]:
                yaw_delta += -self.orbit_key_speed * dt
            if key_states["pitch_up"]:
                pitch_delta += -self.orbit_key_speed * dt
            if key_states["pitch_down"]:
                pitch_delta += +self.orbit_key_speed * dt
            if yaw_delta or pitch_delta:
                self.camera.orbit(yaw_delta, pitch_delta)

        if allow_mouse and imgui.is_mouse_dragging(self._mouse_button, 0.0):
            if not self._mouse_dragging:
                self._mouse_dragging = True
                self.log("mouse drag start")
            drag_delta = imgui.get_mouse_drag_delta(self._mouse_button, 0.0)
            dx = getattr(drag_delta, "x", drag_delta[0] if isinstance(drag_delta, tuple) else 0.0)
            dy = getattr(drag_delta, "y", drag_delta[1] if isinstance(drag_delta, tuple) else 0.0)
            self.camera.orbit(-dx * self.mouse_sensitivity, +dy * self.mouse_sensitivity)
            imgui.reset_mouse_drag_delta(self._mouse_button)
            self.log(f"mouse drag delta dx={dx:.3f} dy={dy:.3f}")
        else:
            if self._mouse_dragging:
                self._mouse_dragging = False
                self.log("mouse drag end")
