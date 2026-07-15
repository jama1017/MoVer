"""Low-latency, explicit-time PNG capture for an initialized animation page."""

from __future__ import annotations

import io
import logging
import math
from collections.abc import Sequence

import numpy as np
from PIL import Image
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page


LOGGER = logging.getLogger(__name__)
RASTER_CAPTURE_STRATEGIES = {"batched", "sequential"}
MAX_BATCH_FRAMES = 100
MAX_SCREENSHOT_DIMENSION = 16_000
MAX_SCREENSHOT_PIXELS = 8_000_000


class BatchCaptureError(RuntimeError):
    """Raised when an assembled browser batch cannot be sliced exactly."""


def _normalize_dimensions(width: int, height: int) -> tuple[int, int]:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("width and height must be positive integers")
    return width, height


def _normalize_times(seek_times: Sequence[float]) -> list[float]:
    normalized: list[float] = []
    for value in seek_times:
        if isinstance(value, bool):
            raise ValueError("seek_times must contain only finite numbers")
        try:
            seek_time = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "seek_times must contain only finite numbers"
            ) from error
        if not math.isfinite(seek_time):
            raise ValueError("seek_times must contain only finite numbers")
        normalized.append(seek_time)
    return normalized


def _plan_batch_chunk_size(width: int, height: int) -> int:
    """Return the largest frame count with a bounded grid layout."""
    frame_pixels = width * height
    max_columns = MAX_SCREENSHOT_DIMENSION // width
    max_rows = MAX_SCREENSHOT_DIMENSION // height
    if max_columns < 1 or max_rows < 1:
        return 0
    upper_bound = min(
        MAX_BATCH_FRAMES,
        MAX_SCREENSHOT_PIXELS // frame_pixels,
        max_columns * max_rows,
    )
    for frame_count in range(upper_bound, 0, -1):
        if _plan_batch_columns(frame_count, width, height):
            return frame_count
    return 0


def _plan_batch_columns(
    frame_count: int,
    width: int,
    height: int,
    viewport_width: int | None = None,
    viewport_height: int | None = None,
) -> int:
    """Choose a compact valid grid for one known chunk."""
    frame_pixels = width * height
    max_columns = min(frame_count, MAX_SCREENSHOT_DIMENSION // width)
    layouts: list[tuple[int, int, int, int]] = []
    for columns in range(1, max_columns + 1):
        rows = math.ceil(frame_count / columns)
        if (
            rows * height <= MAX_SCREENSHOT_DIMENSION
            and columns * rows * frame_pixels <= MAX_SCREENSHOT_PIXELS
        ):
            largest_side = max(columns * width, rows * height)
            layouts.append(
                (largest_side, columns * rows, -columns, columns)
            )
    if viewport_width is not None and viewport_height is not None:
        viewport_layouts = [
            layout
            for layout in layouts
            if (
                layout[3] * width <= viewport_width
                and math.ceil(frame_count / layout[3]) * height
                <= viewport_height
            )
        ]
        if viewport_layouts:
            return max(layout[3] for layout in viewport_layouts)
    return min(layouts)[3] if layouts else 0


def _decode_rgba(png_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png_bytes)) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0


async def get_batched_capture_support(
    page: Page,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, object]:
    """Return the browser's narrow fast-path eligibility result."""
    try:
        result = await page.evaluate(
            "([width, height]) => getBatchCaptureSupport(width, height)",
            [width, height],
        )
    except PlaywrightError as error:
        return {"supported": False, "reason": str(error)}
    if not isinstance(result, dict):
        return {
            "supported": False,
            "reason": "browser returned an invalid eligibility result",
        }
    return result


async def _set_scene_dimensions(
    scene,
    width: int,
    height: int,
    position_at_origin: bool,
) -> str | None:
    return await scene.evaluate(
        """(element, size) => {
            const originalStyle = element.getAttribute("style");
            element.style.setProperty("width", `${size.width}px`, "important");
            element.style.setProperty("height", `${size.height}px`, "important");
            element.style.setProperty("display", "block", "important");
            element.style.setProperty("margin", "0", "important");
            element.style.setProperty("box-sizing", "border-box", "important");
            element.style.setProperty("flex", "none", "important");
            element.style.setProperty("min-width", `${size.width}px`, "important");
            element.style.setProperty("max-width", `${size.width}px`, "important");
            element.style.setProperty("min-height", `${size.height}px`, "important");
            element.style.setProperty("max-height", `${size.height}px`, "important");
            if (size.positionAtOrigin) {
                element.style.setProperty("position", "fixed", "important");
                element.style.setProperty("left", "0", "important");
                element.style.setProperty("top", "0", "important");
            }
            return originalStyle;
        }""",
        {
            "width": width,
            "height": height,
            "positionAtOrigin": position_at_origin,
        },
    )


async def _restore_scene_style(scene, original_style: str | None) -> None:
    await scene.evaluate(
        """(element, originalStyle) => {
            if (originalStyle === null) {
                element.removeAttribute("style");
            } else {
                element.setAttribute("style", originalStyle);
            }
        }""",
        original_style,
    )


async def _capture_sequential(
    page: Page,
    seek_times: list[float],
    width: int,
    height: int,
    hide_grid: bool,
    omit_background: bool,
) -> list[np.ndarray]:
    scene = page.locator("svg").first
    await scene.wait_for(state="visible")
    capture_started = False
    original_style: str | None = None
    dimensions_set = False
    frames: list[np.ndarray] = []
    try:
        capture_started = await page.evaluate("beginServerDrivenCapture()")
        original_style = await _set_scene_dimensions(
            scene,
            width,
            height,
            position_at_origin=not omit_background,
        )
        dimensions_set = True
        for seek_time in seek_times:
            await page.evaluate("time => seekToTime(time)", seek_time)
            await page.evaluate(
                "() => new Promise(resolve => requestAnimationFrame(() => "
                "requestAnimationFrame(resolve)))"
            )
            await page.evaluate("assertNoLateRootAnimations()")
            if hide_grid:
                frame_style = await scene.get_attribute("style")
                await scene.evaluate(
                    """element => element.style.setProperty(
                        "background-image", "none", "important"
                    )"""
                )
            try:
                if omit_background:
                    await scene.scroll_into_view_if_needed()
                    box = await scene.bounding_box()
                    viewport = page.viewport_size
                    if (
                        box is not None
                        and viewport is not None
                        and box["x"] >= 0
                        and box["y"] >= 0
                        and box["x"] + width <= viewport["width"]
                        and box["y"] + height <= viewport["height"]
                    ):
                        png_bytes = await page.screenshot(
                            type="png",
                            scale="css",
                            omit_background=True,
                            clip={
                                "x": box["x"],
                                "y": box["y"],
                                "width": width,
                                "height": height,
                            },
                        )
                    else:
                        png_bytes = await scene.screenshot(
                            type="png",
                            scale="css",
                            omit_background=True,
                        )
                else:
                    png_bytes = await scene.screenshot(
                        type="png",
                        scale="css",
                        omit_background=False,
                    )
            finally:
                if hide_grid:
                    await _restore_scene_style(scene, frame_style)
            frame = _decode_rgba(png_bytes)
            if frame.shape[0] >= height and frame.shape[1] >= width:
                frame = frame[:height, :width].copy()
            if frame.shape != (height, width, 4):
                raise RuntimeError(
                    "Sequential screenshot dimensions do not match request: "
                    f"expected {(height, width, 4)}, got {frame.shape}"
                )
            frames.append(frame)
        return frames
    finally:
        try:
            if dimensions_set:
                await _restore_scene_style(scene, original_style)
        finally:
            if capture_started:
                await page.evaluate("restoreServerDrivenCapture()")


def _validate_batch_geometry(
    geometry: object,
    width: int,
    height: int,
    frame_count: int,
) -> list[tuple[int, int, int, int]]:
    if not isinstance(geometry, dict):
        raise BatchCaptureError("browser returned invalid batch geometry")
    container = geometry.get("container")
    frames = geometry.get("frames")
    dpr = geometry.get("devicePixelRatio")
    columns = geometry.get("columns")
    rows = geometry.get("rows")
    if (
        not isinstance(container, dict)
        or not isinstance(frames, list)
        or len(frames) != frame_count
        or not isinstance(dpr, (int, float))
        or abs(float(dpr) - 1.0) > 1e-6
        or not isinstance(columns, int)
        or not isinstance(rows, int)
        or columns < 1
        or rows != math.ceil(frame_count / columns)
    ):
        raise BatchCaptureError("browser returned incomplete scale-1 geometry")

    tolerance = 0.05
    expected_container = (0, 0, width * columns, height * rows)
    actual_container = tuple(
        container.get(key) for key in ("x", "y", "width", "height")
    )
    if any(
        not isinstance(actual, (int, float))
        or abs(float(actual) - expected) > tolerance
        for actual, expected in zip(actual_container, expected_container)
    ):
        raise BatchCaptureError(
            "batch container geometry does not match requested dimensions: "
            f"expected {expected_container}, got {actual_container}"
        )

    crop_boxes: list[tuple[int, int, int, int]] = []
    for index, frame in enumerate(frames):
        left = (index % columns) * width
        top = (index // columns) * height
        expected = (left, top, width, height)
        actual = (
            tuple(frame.get(key) for key in ("x", "y", "width", "height"))
            if isinstance(frame, dict)
            else ()
        )
        if len(actual) != 4 or any(
            not isinstance(value, (int, float))
            or abs(float(value) - expected_value) > tolerance
            for value, expected_value in zip(actual, expected)
        ):
            raise BatchCaptureError(
                f"batch frame {index} geometry is {actual}; expected {expected}"
            )
        crop_boxes.append((left, top, left + width, top + height))
    return crop_boxes


async def _capture_batched(
    page: Page,
    seek_times: list[float],
    width: int,
    height: int,
    hide_grid: bool,
    omit_background: bool,
) -> list[np.ndarray]:
    if omit_background:
        LOGGER.warning(
            "Batched capture unavailable: transparent output uses the "
            "high-fidelity sequential path"
        )
        return await _capture_sequential(
            page, seek_times, width, height, hide_grid, omit_background
        )
    support = await get_batched_capture_support(page, width, height)
    if not support.get("supported"):
        reason = str(support.get("reason") or "unknown eligibility failure")
        LOGGER.warning("Batched capture unavailable: %s; using sequential", reason)
        return await _capture_sequential(
            page, seek_times, width, height, hide_grid, omit_background
        )

    chunk_size = _plan_batch_chunk_size(width, height)
    if chunk_size < 2:
        LOGGER.warning(
            "Batched capture unavailable: requested dimensions fit fewer than "
            "two frames; using sequential"
        )
        return await _capture_sequential(
            page, seek_times, width, height, hide_grid, omit_background
        )

    frames: list[np.ndarray] = []
    try:
        for start in range(0, len(seek_times), chunk_size):
            chunk = seek_times[start:start + chunk_size]
            viewport = page.viewport_size
            columns = _plan_batch_columns(
                len(chunk),
                width,
                height,
                viewport["width"] if viewport is not None else None,
                viewport["height"] if viewport is not None else None,
            )
            if columns < 1:
                raise BatchCaptureError("no valid screenshot grid for chunk")
            try:
                try:
                    appended = await page.evaluate(
                        """([times, width, height, hideGrid, columns]) =>
                            seekAndAppendToDomUsingTimes(
                                times, width, height, hideGrid, columns, true
                            )""",
                        [chunk, width, height, hide_grid, columns],
                    )
                except PlaywrightError as error:
                    if "Unsupported batch capture:" not in str(error):
                        raise
                    raise BatchCaptureError(str(error)) from error
                if appended != len(chunk):
                    raise BatchCaptureError(
                        f"browser appended {appended} frames; expected {len(chunk)}"
                    )
                await page.evaluate(
                    "() => new Promise(resolve => requestAnimationFrame(() => "
                    "requestAnimationFrame(resolve)))"
                )
                await page.evaluate("assertNoLateRootAnimations()")
                geometry = await page.evaluate("getBatchCaptureGeometry()")
                crop_boxes = _validate_batch_geometry(
                    geometry, width, height, len(chunk)
                )
                rows = math.ceil(len(chunk) / columns)
                batch_width = width * columns
                batch_height = height * rows
                screenshot_options = {
                    "type": "png",
                    "scale": "css",
                    "omit_background": omit_background,
                    "timeout": 10_000,
                }
                try:
                    if (
                        viewport is not None
                        and batch_width <= viewport["width"]
                        and batch_height <= viewport["height"]
                    ):
                        png_bytes = await page.screenshot(
                            **screenshot_options,
                            clip={
                                "x": 0,
                                "y": 0,
                                "width": batch_width,
                                "height": batch_height,
                            },
                        )
                    else:
                        png_bytes = await page.locator("body").screenshot(
                            **screenshot_options
                        )
                except PlaywrightError as error:
                    raise BatchCaptureError(
                        f"batch screenshot failed: {error}"
                    ) from error
            finally:
                await page.evaluate("resetSeekAndAppend()")

            stack = _decode_rgba(png_bytes)
            expected_shape = (height * rows, width * columns, 4)
            if stack.shape != expected_shape:
                raise BatchCaptureError(
                    "batch screenshot dimensions do not match request: "
                    f"expected {expected_shape}, got {stack.shape}"
                )
            for left, top, right, bottom in crop_boxes:
                frames.append(stack[top:bottom, left:right].copy())
        return frames
    except (BatchCaptureError, OSError) as error:
        LOGGER.warning("Batched capture failed: %s; using sequential", error)
        return await _capture_sequential(
            page, seek_times, width, height, hide_grid, omit_background
        )


async def capture_png_frames_at_times(
    page: Page,
    seek_times: Sequence[float],
    *,
    width: int,
    height: int,
    strategy: str = "batched",
    hide_grid: bool = False,
    omit_background: bool = False,
) -> list[np.ndarray]:
    """Return normalized float32 RGBA frames at ordered explicit times.

    The page must already have initialized and prepared MoVer timeline control.
    Batched capture uses one paint wait and screenshot per bounded chunk, then
    falls back to exact-dimension sequential capture when the page is ineligible.
    """
    width, height = _normalize_dimensions(width, height)
    times = _normalize_times(seek_times)
    if strategy not in RASTER_CAPTURE_STRATEGIES:
        supported = ", ".join(sorted(RASTER_CAPTURE_STRATEGIES))
        raise ValueError(
            f"Unsupported capture strategy: {strategy}. Expected: {supported}"
        )
    if not isinstance(hide_grid, bool) or not isinstance(omit_background, bool):
        raise ValueError("hide_grid and omit_background must be booleans")
    if not times:
        return []

    dpr = await page.evaluate("window.devicePixelRatio")
    if (
        not isinstance(dpr, (int, float))
        or abs(float(dpr) - 1.0) > 1e-6
    ):
        raise ValueError("Stage 6C capture requires devicePixelRatio == 1")
    if strategy == "sequential":
        return await _capture_sequential(
            page, times, width, height, hide_grid, omit_background
        )
    return await _capture_batched(
        page, times, width, height, hide_grid, omit_background
    )
