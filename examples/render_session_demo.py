"""Run Stage 6C/6D capture and save a frame contact sheet."""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mover.converter.raster_capture import get_batched_capture_support
from mover.converter.render_session import RenderSession


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_contact_sheet(
    frames: list[np.ndarray],
    width: int,
    height: int,
) -> Image.Image:
    columns = min(10, max(1, math.ceil(math.sqrt(len(frames)))))
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGBA", (columns * width, rows * height))
    for index, frame in enumerate(frames):
        pixels = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels)
        sheet.paste(
            image,
            ((index % columns) * width, (index // columns) * height),
        )
    return sheet


async def run(args: argparse.Namespace) -> None:
    html_path = args.html_file.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else Path.cwd() / f"{html_path.stem}_render_session_contact_sheet.png"
    )
    session = RenderSession(
        html_path,
        browser_launch_options={"headless": not args.headed},
    )

    startup_started = time.perf_counter()
    await session.start()
    startup_s = time.perf_counter() - startup_started
    try:
        info = await session.get_animation_info()
        duration = float(info["animDuration"])
        times = np.linspace(0.0, duration, args.frames).tolist()
        support = await get_batched_capture_support(
            session.page,
            args.width,
            args.height,
        )
        navigation_count = await session.evaluate(
            "performance.getEntriesByType('navigation').length"
        )

        samples: list[float] = []
        first_frames: list[np.ndarray] | None = None
        final_frames: list[np.ndarray] = []
        for _ in range(args.runs):
            capture_started = time.perf_counter()
            final_frames = await session.capture(
                times,
                width=args.width,
                height=args.height,
            )
            samples.append(time.perf_counter() - capture_started)
            if first_frames is None:
                first_frames = final_frames

        assert first_frames is not None
        repeat_delta = float(
            np.abs(
                np.stack(first_frames) - np.stack(final_frames)
            ).max()
            * 255.0
        )
        navigation_preserved = (
            await session.evaluate(
                "performance.getEntriesByType('navigation').length"
            )
            == navigation_count
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        build_contact_sheet(
            final_frames,
            args.width,
            args.height,
        ).save(output_path)

        route = "batched" if support.get("supported") else "sequential fallback"
        print(f"Source: {html_path}")
        print(f"Animation duration: {duration:.3f}s")
        print(f"Capture route: {route}")
        if not support.get("supported"):
            print(f"Fallback reason: {support.get('reason')}")
        print(f"Session startup: {startup_s * 1000:.1f}ms")
        print(
            f"Warm captures: {np.mean(samples) * 1000:.1f}ms mean "
            f"over {len(samples)} run(s)"
        )
        print(f"Repeated-run max delta: {repeat_delta:.1f}/255")
        print(f"Page navigation unchanged: {navigation_preserved}")
        print(f"Contact sheet: {output_path}")

        if args.headed and args.hold_seconds > 0:
            print(
                f"Keeping the browser open for {args.hold_seconds:.1f}s..."
            )
            await asyncio.sleep(args.hold_seconds)
    finally:
        await session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture repeated frame batches through one persistent RenderSession "
            "and save the final frames as a contact sheet."
        )
    )
    parser.add_argument(
        "html_file",
        type=Path,
        help="Animation HTML file to serve and capture",
    )
    parser.add_argument("--width", type=positive_int, default=128)
    parser.add_argument("--height", type=positive_int, default=128)
    parser.add_argument("--frames", type=positive_int, default=30)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Contact-sheet PNG path",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the Chromium window",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Keep a headed browser open after capture",
    )
    args = parser.parse_args()
    if args.hold_seconds < 0:
        parser.error("--hold-seconds must be non-negative")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
