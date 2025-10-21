from __future__ import annotations

import json
import os
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state

from viewer import run_viewer


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="GS-IR PBR imgui_bundle Viewer")
    parser.add_argument("--config", type=str, default="config.json", help="Path to a JSON configuration file.")
    ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--checkpoint", type=str, nargs="+", help="Path(s) to GS-IR checkpoint(s) to load.")
    parser.add_argument("--hdri", type=str, default=None, help="Path to a specific HDRI file (.hdr).")
    parser.add_argument("--hdri_root", type=str, default=None, help="Directory containing HDRI preset files.")
    parser.add_argument("--env_res", type=int, default=256, help="Cubemap base resolution per face.")
    parser.add_argument("--width", type=int, default=1280, help="Viewer window width.")
    parser.add_argument("--height", type=int, default=720, help="Viewer window height.")
    parser.add_argument("--tone", action="store_true", help="Enable ACES filmic tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear->sRGB gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Use reconstructed metallic map.")
    parser.add_argument("--no_env_bg", action="store_true", help="Disable compositing HDRI as background.")
    parser.add_argument(
        "--transform_state",
        type=str,
        default="viewer_transforms.json",
        help="Path to store/load per-model transforms.",
    )
    parser.set_defaults(pipeline_spec=pipeline)
    return parser


def load_config(parser: ArgumentParser):
    temp_args, _ = parser.parse_known_args()
    if temp_args.config and os.path.isfile(temp_args.config):
        print(f"[viewer] Loading arguments from: {temp_args.config}")
        with open(temp_args.config, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)
        parser.set_defaults(**config_data)
    else:
        print(f"[viewer] Config file not found at '{temp_args.config}'. Using command-line arguments and defaults.")


def main() -> None:
    parser = build_parser()
    load_config(parser)
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("A --checkpoint must be provided.")
    if not args.hdri and not args.hdri_root:
        parser.error("Provide --hdri (file) or --hdri_root (directory).")

    pipeline_spec: PipelineParams = args.pipeline_spec
    pipeline = pipeline_spec.extract(args)

    safe_state(getattr(args, "quiet", False))

    arg_obj = type("Args", (), {})()
    arg_obj.__dict__.update(vars(args))
    arg_obj.pipeline = pipeline
    arg_obj.sh_degree = args.sh_degree

    run_viewer(arg_obj)


if __name__ == "__main__":
    main()
