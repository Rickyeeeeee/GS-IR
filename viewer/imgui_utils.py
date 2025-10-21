from __future__ import annotations

from imgui_bundle import imgui


def name_variants(name: str) -> list[str]:
    if not name:
        return []
    variants = {name, name.lower(), name.upper()}
    snake = []
    current = []
    for idx, ch in enumerate(name):
        if ch.isupper() and idx > 0 and not name[idx - 1].isupper():
            current.append("_")
        current.append(ch.lower())
    snake_name = "".join(current)
    if snake_name:
        snake.append(snake_name)
    camel_alt = name.replace(" ", "_").replace("-", "_")
    variants.update({camel_alt, camel_alt.lower(), camel_alt.upper(), *snake})
    return [v for v in variants if v]


def try_resolve_imgui_constant(family: str, *names: str):
    containers_map = {
        "key": ["Key", "ImGuiKey"],
        "mouse_button": ["MouseButton", "MouseButton_", "ImGuiMouseButton"],
        "cond": ["Cond", "Cond_", "ImGuiCond"],
        "hovered": ["HoveredFlags", "HoveredFlags_", "ImGuiHoveredFlags"],
    }
    containers = containers_map.get(family, [])

    for container_name in containers:
        container = getattr(imgui, container_name, None)
        if container is None:
            continue
        members = getattr(container, "__members__", None)
        for original in names:
            for variant in name_variants(original):
                if members and variant in members:
                    return members[variant]
                value = getattr(container, variant, None)
                if value is not None:
                    return value

    for name in names:
        for variant in name_variants(name):
            value = getattr(imgui, variant, None)
            if value is not None:
                return value
    return None


def imgui_key(name: str):
    return try_resolve_imgui_constant("key", name, name.upper(), name.lower(), f"Key_{name}", f"key_{name}")


def imgui_cond(name: str):
    return try_resolve_imgui_constant("cond", name, name.capitalize(), name.lower())


def imgui_mouse_button(name: str):
    return try_resolve_imgui_constant("mouse_button", name, name.capitalize(), name.lower())


def imgui_hovered_flag(name: str, default: int = 0) -> int:
    value = try_resolve_imgui_constant("hovered", name, name.capitalize(), name.lower())
    return int(value) if value is not None else default


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
