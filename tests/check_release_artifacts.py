#!/usr/bin/env python3
"""Validate MoVer wheel/sdist contents and rebuilt-wheel parity."""

from __future__ import annotations

import argparse
import email
import fnmatch
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path


REQUIRED_RESOURCES = {
    "mover/converter/assets/EasePack.min.js",
    "mover/converter/assets/MotionPathPlugin.min.js",
    "mover/converter/assets/api.js",
    "mover/converter/assets/convert.js",
    "mover/converter/assets/grid.svg",
    "mover/converter/assets/gsap.min.js",
    "mover/converter/assets/index.css",
    "mover/converter/assets/property_registry.json",
    "mover/converter/assets/vis.js",
    "mover/dsl/assets/correction_msg_template.md",
    "mover/nlg/assets/sentence_patterns.json",
    "mover/nlg/assets/vocab.json",
    "mover/synthesizers/assets/sys_msg_animation_synthesizer.md",
    "mover/synthesizers/assets/sys_msg_animation_synthesizer_with_implementation.md",
    "mover/synthesizers/assets/sys_msg_mover_synthesizer.md",
    "mover/synthesizers/assets/sys_msg_prompt_rewriter.md",
    "mover/synthesizers/assets/template.html",
    "mover/_vendor/concepts/LICENSE",
    "mover/_vendor/concepts/VENDORED.md",
    "mover/_vendor/torch_index/LICENSE",
    "mover/_vendor/torch_index/VENDORED.md",
}

FORBIDDEN_WHEEL_PREFIXES = (
    "examples/",
    "mover/composers/",
    "mover_dataset/",
    "plans/",
    "test_output/",
    "tests/",
)

FORBIDDEN_SDIST_PREFIXES = ("plans/", "test_output/")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_wheel(path: Path) -> tuple[set[str], dict[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        members = {name for name in archive.namelist() if not name.endswith("/")}
        contents = {name: archive.read(name) for name in members}
    return members, contents


def read_sdist(path: Path) -> tuple[set[str], dict[str, bytes]]:
    with tarfile.open(path, "r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        roots = {member.name.split("/", 1)[0] for member in files}
        require(len(roots) == 1, f"Unexpected sdist roots: {roots}")
        root = next(iter(roots))
        contents = {}
        for member in files:
            relative = member.name.removeprefix(f"{root}/")
            extracted = archive.extractfile(member)
            require(extracted is not None, f"Unable to read {member.name}")
            contents[relative] = extracted.read()
    return set(contents), contents


def metadata_from_wheel(contents: dict[str, bytes]):
    paths = [name for name in contents if name.endswith(".dist-info/METADATA")]
    require(len(paths) == 1, f"Expected one wheel METADATA file, found {paths}")
    return email.message_from_bytes(contents[paths[0]])


def metadata_from_sdist(contents: dict[str, bytes]):
    require("PKG-INFO" in contents, "sdist is missing PKG-INFO")
    return email.message_from_bytes(contents["PKG-INFO"])


def normalized_metadata(message) -> dict[str, object]:
    return {
        "Name": message["Name"],
        "Version": message["Version"],
        "Requires-Python": message["Requires-Python"],
        "Requires-Dist": sorted(message.get_all("Requires-Dist", [])),
        "Provides-Extra": sorted(message.get_all("Provides-Extra", [])),
    }


def check_wheel(path: Path) -> tuple[set[str], dict[str, bytes], dict[str, object]]:
    members, contents = read_wheel(path)
    missing = sorted(REQUIRED_RESOURCES - members)
    require(not missing, f"Wheel is missing required resources: {missing}")

    forbidden = sorted(
        name
        for name in members
        if name.startswith(FORBIDDEN_WHEEL_PREFIXES)
        or fnmatch.fnmatch(name, "mover/converter/assets/library*.js")
        or name.endswith("/sys_msg_test.md")
    )
    require(not forbidden, f"Forbidden wheel members: {forbidden}")
    require(
        "mover/converter/mover_querier.py" in members,
        "Wheel is missing the experimental module-only querier",
    )

    entry_points = [
        name for name in members if name.endswith(".dist-info/entry_points.txt")
    ]
    require(len(entry_points) == 1, "Wheel is missing entry_points.txt")
    require(
        b"mover-convert = mover.converter.mover_converter:main"
        in contents[entry_points[0]],
        "Wheel has an incorrect mover-convert entry point",
    )

    metadata = normalized_metadata(metadata_from_wheel(contents))
    require(metadata["Name"] == "mover", metadata)
    require(metadata["Version"] == "0.3.1", metadata)
    require(metadata["Requires-Python"] == "<3.13,>=3.10", metadata)
    for extra in ("full", "groq", "media", "ollama", "openai", "vertex"):
        require(extra in metadata["Provides-Extra"], f"Missing extra: {extra}")
    return members, contents, metadata


def check_sdist(
    path: Path,
    wheel_metadata: dict[str, object],
) -> tuple[set[str], dict[str, bytes]]:
    members, contents = read_sdist(path)
    for required in ("LICENSE", "README.md", "pyproject.toml", "PKG-INFO"):
        require(required in members, f"sdist is missing {required}")
    for resource in REQUIRED_RESOURCES:
        require(f"src/{resource}" in members, f"sdist is missing src/{resource}")

    forbidden = sorted(
        name
        for name in members
        if name.startswith(FORBIDDEN_SDIST_PREFIXES)
        or fnmatch.fnmatch(name, "src/mover/converter/assets/library*.js")
    )
    require(not forbidden, f"Forbidden sdist members: {forbidden}")

    sdist_metadata = normalized_metadata(metadata_from_sdist(contents))
    require(
        sdist_metadata == wheel_metadata,
        f"Wheel/sdist metadata differs: {wheel_metadata} != {sdist_metadata}",
    )
    return members, contents


def comparable_wheel_contents(contents: dict[str, bytes]) -> dict[str, str]:
    comparable = {}
    for name, data in contents.items():
        if name.endswith(".dist-info/RECORD"):
            continue
        comparable[name] = hashlib.sha256(data).hexdigest()
    return comparable


def check_rebuilt_wheel(
    direct_path: Path,
    direct_members: set[str],
    direct_contents: dict[str, bytes],
    rebuilt_path: Path,
) -> None:
    rebuilt_members, rebuilt_contents, rebuilt_metadata = check_wheel(rebuilt_path)
    direct_metadata = normalized_metadata(metadata_from_wheel(direct_contents))
    require(rebuilt_metadata == direct_metadata, "Rebuilt wheel metadata differs")
    require(
        rebuilt_members == direct_members,
        "Direct and sdist-rebuilt wheel manifests differ",
    )
    require(
        comparable_wheel_contents(rebuilt_contents)
        == comparable_wheel_contents(direct_contents),
        "Direct and sdist-rebuilt wheel payload hashes differ",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--rebuilt-wheel", type=Path)
    args = parser.parse_args()

    direct_members, direct_contents, wheel_metadata = check_wheel(args.wheel)
    sdist_members, _ = check_sdist(args.sdist, wheel_metadata)
    if args.rebuilt_wheel:
        check_rebuilt_wheel(
            args.wheel,
            direct_members,
            direct_contents,
            args.rebuilt_wheel,
        )

    print(
        json.dumps(
            {
                "wheel": str(args.wheel),
                "wheel_files": len(direct_members),
                "sdist": str(args.sdist),
                "sdist_files": len(sdist_members),
                "rebuilt_wheel": (
                    str(args.rebuilt_wheel) if args.rebuilt_wheel else None
                ),
                "metadata": wheel_metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
