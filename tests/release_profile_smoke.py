#!/usr/bin/env python3
"""Smoke-test an installed MoVer release profile outside the source checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import sys
import tempfile
from importlib import resources
from pathlib import Path


CONVERTER_ASSETS = {
    "EasePack.min.js",
    "MotionPathPlugin.min.js",
    "api.js",
    "convert.js",
    "grid.svg",
    "gsap.min.js",
    "index.css",
    "property_registry.json",
    "vis.js",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def check_common(repo_root: Path) -> dict[str, object]:
    mover = importlib.import_module("mover")
    package_root = Path(mover.__file__).resolve().parent
    require(
        repo_root.resolve() not in package_root.parents,
        f"MoVer imported from source checkout: {package_root}",
    )
    require(
        importlib.metadata.version("mover") == "0.2.0",
        "Installed distribution is not mover 0.2.0",
    )

    converter_assets = resources.files("mover.converter").joinpath("assets")
    for name in CONVERTER_ASSETS:
        require(converter_assets.joinpath(name).is_file(), f"Missing asset: {name}")
    require(
        not converter_assets.joinpath("library.js").is_file(),
        "Ignored library.js entered the installed wheel",
    )
    require(
        not converter_assets.joinpath("library_raw.js").is_file(),
        "Ignored library_raw.js entered the installed wheel",
    )
    require(
        not resources.files("mover.synthesizers")
        .joinpath("assets", "sys_msg_test.md")
        .is_file(),
        "Development-only sys_msg_test.md entered the installed wheel",
    )
    require(
        not module_available("mover.composers"),
        "Excluded mover.composers package entered the installed wheel",
    )

    entry_points = [
        entry_point
        for entry_point in importlib.metadata.entry_points(group="console_scripts")
        if entry_point.name == "mover-convert"
    ]
    require(len(entry_points) == 1, "mover-convert entry point is missing")
    require(
        entry_points[0].value == "mover.converter.mover_converter:main",
        f"Unexpected mover-convert target: {entry_points[0].value}",
    )

    return {"package_root": str(package_root)}


def check_minimal() -> dict[str, object]:
    absent = ("cv2", "groq", "jacinle", "openai", "pyrealb", "torch")
    unexpected = [name for name in absent if module_available(name)]
    require(not unexpected, f"Minimal profile contains full/media packages: {unexpected}")

    importlib.import_module("mover.converter.mover_converter")
    try:
        importlib.import_module("mover.pipeline")
    except ModuleNotFoundError as error:
        require(
            'pip install "mover[full]"' in str(error),
            f"Pipeline error is not actionable: {error}",
        )
    else:
        raise AssertionError("Minimal profile unexpectedly imported mover.pipeline")

    return {"absent_modules": list(absent)}


def check_full(repo_root: Path) -> dict[str, object]:
    required = (
        "cv2",
        "groq",
        "jacinle",
        "jinja2",
        "lark",
        "openai",
        "pyrealb",
        "torch",
        "treelib",
        "yaml",
    )
    missing = [name for name in required if not module_available(name)]
    require(not missing, f"Full profile is missing dependencies: {missing}")

    development_only = ("ipykernel", "tree_sitter", "tree_sitter_language_pack")
    leaked = [name for name in development_only if module_available(name)]
    require(not leaked, f"Development-only packages entered full profile: {leaked}")

    importlib.import_module("mover.pipeline")
    importlib.import_module("mover.nlg.prompt_generator")
    importlib.import_module("mover.synthesizers.animation_synthesizer")
    llm_client = importlib.import_module("mover.synthesizers.llm_client")
    providers = llm_client.get_available_providers()
    require("openai" in providers and "groq" in providers, providers)

    from mover.dsl.mover_verifier import MoverVerifier

    result = MoverVerifier().verify(
        str(repo_root / "examples" / "translate_right.py"),
        str(repo_root / "examples" / "translate_right_data.json"),
    )
    require(result.strip().endswith("agent: True"), result)

    sys.path.insert(0, str(repo_root / "mover_dataset"))
    try:
        from dataset_scene_graphs import gen_data_all
        from mover.nlg.prompt_generator import PromptGenerator

        prompts = PromptGenerator().generate(gen_data_all[0])
    finally:
        sys.path.pop(0)
    require(prompts, "Offline prompt generation returned no prompts")

    return {"available_providers": providers, "prompt_count": len(prompts)}


def check_ffmpeg_media(ffmpeg_executable: Path | None) -> dict[str, object]:
    import numpy as np

    import mover.converter.mover_converter as converter

    original_run = converter.subprocess.run
    if ffmpeg_executable is not None:
        require(ffmpeg_executable.is_file(), f"Missing FFmpeg: {ffmpeg_executable}")

        def run_with_selected_ffmpeg(command, *args, **kwargs):
            if command[0] == "ffmpeg":
                command = [str(ffmpeg_executable), *command[1:]]
            return original_run(command, *args, **kwargs)

        converter.subprocess.run = run_with_selected_ffmpeg

    frames = [
        np.full((16, 16, 3), fill_value=index * 20, dtype=np.uint8)
        for index in range(6)
    ]
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mp4_path = root / "animation.mp4"
            gif_path = root / "animation.gif"
            converter.create_video_from_frames(frames, str(mp4_path), 6, "mp4")
            converter.create_video_from_frames(frames, str(gif_path), 6, "gif")
            require(mp4_path.stat().st_size > 0, "FFmpeg produced no MP4")
            require(gif_path.stat().st_size > 0, "FFmpeg produced no GIF")
            return {
                "mp4_bytes": mp4_path.stat().st_size,
                "gif_bytes": gif_path.stat().st_size,
            }
    finally:
        converter.subprocess.run = original_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("minimal", "full"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--ffmpeg-executable", type=Path)
    args = parser.parse_args()

    summary = {"profile": args.profile}
    summary.update(check_common(args.repo_root))
    if args.profile == "minimal":
        summary.update(check_minimal())
    else:
        summary.update(check_full(args.repo_root))
    summary["media"] = check_ffmpeg_media(args.ffmpeg_executable)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
