# test_imgui_bundle.py
from imgui_bundle import imgui, immapp, hello_imgui

def gui():
    imgui.text("Hello, ImGui Bundle 1.3.0!")
    imgui.separator()

    if imgui.button("Click me!"):
        print("Button clicked!")

    _, value = imgui.slider_float("Slider", 0.5, 0.0, 1.0)
    imgui.text(f"Slider value: {value:.2f}")

def main():

    # app_settings.docking_params.enable_docking = True

    immapp.run(
        gui_function=gui,
        window_size=(1200, 900),
        window_title="Testing"
    )

if __name__ == "__main__":
    main()

# imgui_bundle version: 1.3.0