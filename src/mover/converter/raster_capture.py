"""Resolution-controlled sequential and batched PNG capture."""

from __future__ import annotations

import io
import math
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page


RASTER_CAPTURE_STRATEGIES = {"sequential", "batched"}
_MAX_BATCH_SCREENSHOT_DIMENSION = 16_000
_MAX_BATCH_SCREENSHOT_PIXELS = 8_000_000
_MAX_BATCH_FRAMES_PER_SCREENSHOT = 100


def _normalize_raster_dimensions(
    width: int | None,
    height: int | None,
) -> tuple[int, int] | None:
    """Validate optional paired logical raster dimensions."""
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise ValueError("width and height must be provided together")
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


def _normalize_capture_strategy(value: str) -> str:
    """Return a supported raster capture strategy."""
    if not isinstance(value, str):
        raise ValueError("capture strategy must be a string")
    strategy = value.lower()
    if strategy not in RASTER_CAPTURE_STRATEGIES:
        supported = ", ".join(sorted(RASTER_CAPTURE_STRATEGIES))
        raise ValueError(
            f"Unsupported capture strategy: {value}. Expected one of: {supported}"
        )
    return strategy


def _normalize_seek_times(seek_times: Sequence[float]) -> list[float]:
    """Copy and validate ordered explicit capture times."""
    normalized: list[float] = []
    for value in seek_times:
        if isinstance(value, bool):
            raise ValueError("seek_times must contain only finite numbers")
        try:
            time_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "seek_times must contain only finite numbers"
            ) from error
        if not math.isfinite(time_value):
            raise ValueError("seek_times must contain only finite numbers")
        normalized.append(time_value)
    return normalized


def _plan_batch_layout(
    width: int,
    height: int,
    viewport_width: int,
    viewport_height: int,
) -> tuple[int, int]:
    """Return ``(frame_capacity, columns)`` for one bounded grid."""
    frame_pixels = width * height
    if (
        width > _MAX_BATCH_SCREENSHOT_DIMENSION
        or height > _MAX_BATCH_SCREENSHOT_DIMENSION
        or frame_pixels > _MAX_BATCH_SCREENSHOT_PIXELS
    ):
        return 0, 0
    max_columns = min(
        viewport_width // width,
        _MAX_BATCH_SCREENSHOT_DIMENSION // width,
        _MAX_BATCH_FRAMES_PER_SCREENSHOT,
    )
    max_rows = min(
        viewport_height // height,
        _MAX_BATCH_SCREENSHOT_DIMENSION // height,
        _MAX_BATCH_FRAMES_PER_SCREENSHOT,
    )
    best_capacity = 0
    best_columns = 0
    for columns in range(1, max_columns + 1):
        rows = min(
            max_rows,
            _MAX_BATCH_SCREENSHOT_PIXELS // (columns * frame_pixels),
        )
        if rows < 1:
            continue
        capacity = min(
            _MAX_BATCH_FRAMES_PER_SCREENSHOT,
            columns * rows,
        )
        if capacity > best_capacity:
            best_capacity = capacity
            best_columns = columns
    return best_capacity, best_columns


def _plan_batch_chunk_size(width: int, height: int) -> int:
    """Return the global maximum frame count for compatibility/tests."""
    capacity, _columns = _plan_batch_layout(
        width,
        height,
        _MAX_BATCH_SCREENSHOT_DIMENSION,
        _MAX_BATCH_SCREENSHOT_DIMENSION,
    )
    return capacity


async def _capture_svg_png(
    svg_element,
    hide_grid: bool,
    *,
    scale: str | None = None,
    omit_background: bool = False,
    page: Page | None = None,
    clip: dict[str, int] | None = None,
) -> bytes:
    screenshot_options: dict[str, object] = {"type": "png"}
    if scale is not None:
        screenshot_options["scale"] = scale
    if omit_background:
        screenshot_options["omit_background"] = True
    if clip is not None:
        if page is None:
            raise ValueError("page is required when screenshot clip is set")
        screenshot_options["clip"] = clip
    if not hide_grid:
        if clip is not None:
            return await page.screenshot(**screenshot_options)
        return await svg_element.screenshot(**screenshot_options)

    original_style = await svg_element.get_attribute("style")
    await svg_element.evaluate(
        """element =>
            element.style.setProperty("background-image", "none", "important")
        """
    )
    try:
        if clip is not None:
            return await page.screenshot(**screenshot_options)
        return await svg_element.screenshot(**screenshot_options)
    finally:
        await svg_element.evaluate(
            """(element, originalStyle) => {
                if (originalStyle === null) {
                    element.removeAttribute("style");
                } else {
                    element.setAttribute("style", originalStyle);
                }
            }""",
            original_style,
        )


class _BatchGeometryError(RuntimeError):
    """Raised when a browser batch cannot be cropped exactly."""


async def _set_svg_capture_dimensions(
    svg_element,
    dimensions: tuple[int, int] | None,
) -> str | None:
    if dimensions is None:
        return None
    width, height = dimensions
    return await svg_element.evaluate(
        """(element, dimensions) => {
            const originalStyle = element.getAttribute("style");
            element.style.setProperty(
                "width",
                `${dimensions.width}px`,
                "important"
            );
            element.style.setProperty(
                "height",
                `${dimensions.height}px`,
                "important"
            );
            element.style.setProperty("display", "block", "important");
            element.style.setProperty("position", "fixed", "important");
            element.style.setProperty("left", "0", "important");
            element.style.setProperty("top", "0", "important");
            element.style.setProperty("margin", "0", "important");
            element.style.setProperty("box-sizing", "border-box", "important");
            return originalStyle;
        }""",
        {"width": width, "height": height},
    )


async def _restore_svg_capture_dimensions(
    svg_element,
    original_style: str | None,
) -> None:
    await svg_element.evaluate(
        """(element, originalStyle) => {
            if (originalStyle === null) {
                element.removeAttribute("style");
            } else {
                element.setAttribute("style", originalStyle);
            }
        }""",
        original_style,
    )


def _validate_batch_geometry(
    geometry: dict,
    width: int,
    height: int,
    frame_count: int,
    columns: int,
) -> list[tuple[int, int, int, int]]:
    """Validate measured CSS bounds and return crop boxes."""
    tolerance = 0.05
    container = geometry.get("container")
    frames = geometry.get("frames")
    dpr = geometry.get("devicePixelRatio")
    if (
        not isinstance(container, dict)
        or not isinstance(frames, list)
        or len(frames) != frame_count
        or not isinstance(dpr, (int, float))
        or not math.isfinite(float(dpr))
        or abs(float(dpr) - 1.0) > 1e-6
    ):
        raise _BatchGeometryError(
            "Browser returned incomplete geometry or non-scale-1 DPR"
        )

    effective_columns = min(columns, frame_count)
    rows = math.ceil(frame_count / effective_columns)
    expected_container = (width * effective_columns, height * rows)
    actual_container = (container.get("width"), container.get("height"))
    if any(
        not isinstance(actual, (int, float))
        or abs(float(actual) - expected) > tolerance
        for actual, expected in zip(actual_container, expected_container)
    ):
        raise _BatchGeometryError(
            "Batch container geometry does not match requested dimensions: "
            f"expected {expected_container}, got {actual_container}"
        )

    crop_boxes: list[tuple[int, int, int, int]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise _BatchGeometryError(f"Missing geometry for batch frame {index}")
        frame_values = (
            frame.get("x"),
            frame.get("y"),
            frame.get("width"),
            frame.get("height"),
            container.get("x"),
            container.get("y"),
        )
        if any(not isinstance(value, (int, float)) for value in frame_values):
            raise _BatchGeometryError(
                f"Missing numeric geometry for batch frame {index}"
            )
        relative_x = frame_values[0] - frame_values[4]
        relative_y = frame_values[1] - frame_values[5]
        expected = (
            (index % effective_columns) * width,
            (index // effective_columns) * height,
            width,
            height,
        )
        actual = (
            relative_x,
            relative_y,
            frame.get("width"),
            frame.get("height"),
        )
        if any(
            not isinstance(value, (int, float))
            or abs(float(value) - expected_value) > tolerance
            for value, expected_value in zip(actual, expected)
        ):
            raise _BatchGeometryError(
                f"Batch frame {index} geometry does not match requested bounds: "
                f"expected {expected}, got {actual}"
            )
        left = round(relative_x)
        top = round(relative_y)
        crop_boxes.append((left, top, left + width, top + height))
    return crop_boxes


async def _capture_png_frames_sequential_at_times(
    page: Page,
    seek_times: list[float],
    dimensions: tuple[int, int] | None,
    hide_grid: bool,
    omit_background: bool,
    frame_sink: Callable[[int, np.ndarray], None] | None = None,
) -> list[np.ndarray]:
    svg_element = page.locator("svg").first
    await svg_element.wait_for(state="visible")
    original_style: str | None = None
    original_viewport = page.viewport_size
    capture_viewport = original_viewport
    dimensions_applied = False
    capture_started = False
    frames: list[np.ndarray] = []
    try:
        capture_started = await page.evaluate("beginServerDrivenCapture()")
        if dimensions is not None:
            if original_viewport is not None:
                # Chromium screenshot capture can stall on very small
                # viewports. Expand only for painting; the requested SVG box
                # and returned pixels remain exact, and the viewport is
                # restored immediately after each screenshot.
                capture_viewport = {
                    "width": max(
                        original_viewport["width"],
                        dimensions[0],
                        128,
                    ),
                    "height": max(
                        original_viewport["height"],
                        dimensions[1],
                        128,
                    ),
                }
            original_style = await _set_svg_capture_dimensions(
                svg_element,
                dimensions,
            )
            dimensions_applied = True
        for frame_index, seek_time in enumerate(seek_times):
            await page.evaluate("time => seekToTime(time)", seek_time)
            await page.evaluate(
                "() => new Promise(r => requestAnimationFrame(() => "
                "requestAnimationFrame(r)))"
            )
            await page.evaluate("assertNoLateRootAnimations()")
            viewport_changed = False
            try:
                if (
                    capture_viewport is not None
                    and capture_viewport != original_viewport
                ):
                    await page.set_viewport_size(capture_viewport)
                    viewport_changed = True
                png_bytes = await _capture_svg_png(
                    svg_element,
                    hide_grid,
                    scale="css",
                    omit_background=omit_background,
                    page=page,
                    clip=(
                        {
                            "x": 0,
                            "y": 0,
                            "width": dimensions[0],
                            "height": dimensions[1],
                        }
                        if dimensions is not None
                        else None
                    ),
                )
            finally:
                if viewport_changed and original_viewport is not None:
                    await page.set_viewport_size(original_viewport)
            image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            if dimensions is not None and image.size != dimensions:
                raise RuntimeError(
                    "Sequential screenshot dimensions do not match request: "
                    f"expected {dimensions}, got {image.size}"
                )
            frame_pixels = np.asarray(image, dtype=np.uint8)
            if frame_sink is None:
                frames.append(
                    frame_pixels.astype(np.float32) / 255.0
                )
            else:
                frame_sink(frame_index, frame_pixels)
        return frames
    finally:
        try:
            try:
                if (
                    original_viewport is not None
                    and page.viewport_size != original_viewport
                ):
                    await page.set_viewport_size(original_viewport)
            finally:
                if dimensions_applied:
                    await _restore_svg_capture_dimensions(
                        svg_element,
                        original_style,
                    )
        finally:
            if capture_started:
                await page.evaluate("restoreServerDrivenCapture()")


async def _capture_png_frames_batched_at_times(
    page: Page,
    seek_times: list[float],
    dimensions: tuple[int, int],
    hide_grid: bool,
    omit_background: bool,
    frame_sink: Callable[[int, np.ndarray], None] | None = None,
) -> list[np.ndarray]:
    width, height = dimensions
    if len(seek_times) < 2:
        return await _capture_png_frames_sequential_at_times(
            page,
            seek_times,
            dimensions,
            hide_grid,
            omit_background,
            frame_sink,
        )
    viewport = page.viewport_size
    try:
        support = await page.evaluate("getBatchCaptureSupport()")
    except PlaywrightError as error:
        support = {"supported": False, "reason": str(error)}
    if not support.get("supported"):
        print(
            f"Batch capture is unsupported ({support.get('reason')}); "
            "falling back to sequential capture"
        )
        return await _capture_png_frames_sequential_at_times(
            page,
            seek_times,
            dimensions,
            hide_grid,
            omit_background,
            frame_sink,
        )
    if viewport is None:
        print(
            "Batch capture requires a fixed viewport; falling back to "
            "sequential capture"
        )
        return await _capture_png_frames_sequential_at_times(
            page,
            seek_times,
            dimensions,
            hide_grid,
            omit_background,
            frame_sink,
        )

    chunk_size, columns = _plan_batch_layout(
        width,
        height,
        viewport["width"],
        viewport["height"],
    )
    if chunk_size < 2:
        print(
            "The current viewport fits fewer than two requested frames; "
            "falling back to sequential capture"
        )
        return await _capture_png_frames_sequential_at_times(
            page,
            seek_times,
            dimensions,
            hide_grid,
            omit_background,
            frame_sink,
        )
    chunk_count = math.ceil(len(seek_times) / chunk_size)
    minimum_chunk_size = len(seek_times) // chunk_count
    if minimum_chunk_size < 2:
        print(
            "The request cannot be split without a singleton remainder; "
            "falling back to sequential capture"
        )
        return await _capture_png_frames_sequential_at_times(
            page,
            seek_times,
            dimensions,
            hide_grid,
            omit_background,
            frame_sink,
        )
    larger_chunk_count = len(seek_times) % chunk_count
    chunk_sizes = [
        minimum_chunk_size + (1 if index < larger_chunk_count else 0)
        for index in range(chunk_count)
    ]
    columns = min(columns, max(chunk_sizes))

    frames: list[np.ndarray] = []
    start = 0
    try:
        for current_chunk_size in chunk_sizes:
            chunk = seek_times[start:start + current_chunk_size]
            effective_columns = min(columns, len(chunk))
            try:
                try:
                    appended = await page.evaluate(
                        """([times, width, height, hideGrid, columns]) =>
                            seekAndAppendToDomUsingTimes(
                                times,
                                width,
                                height,
                                hideGrid,
                                columns
                            )""",
                        [chunk, width, height, hide_grid, effective_columns],
                    )
                except PlaywrightError as error:
                    if "Unsupported batch capture:" not in str(error):
                        raise
                    raise _BatchGeometryError(str(error)) from error
                if appended != len(chunk):
                    raise _BatchGeometryError(
                        f"Batch appended {appended} frames; expected {len(chunk)}"
                    )
                try:
                    geometry = await page.evaluate(
                        "getBatchCaptureGeometry()"
                    )
                    crop_boxes = _validate_batch_geometry(
                        geometry,
                        width,
                        height,
                        len(chunk),
                        effective_columns,
                    )
                    container = geometry["container"]
                    png_bytes = await page.screenshot(
                        type="png",
                        scale="css",
                        omit_background=omit_background,
                        clip={
                            "x": container["x"],
                            "y": container["y"],
                            "width": container["width"],
                            "height": container["height"],
                        },
                        timeout=10_000,
                    )
                except PlaywrightError as error:
                    raise _BatchGeometryError(
                        f"Batch screenshot failed: {error}"
                    ) from error
                image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                expected_size = (
                    width * effective_columns,
                    height * math.ceil(len(chunk) / effective_columns),
                )
                if image.size != expected_size:
                    raise _BatchGeometryError(
                        "Batch screenshot dimensions do not match measured "
                        f"container: expected {expected_size}, got {image.size}"
                    )
                stack = np.asarray(image, dtype=np.uint8)
                for chunk_index, (left, top, right, bottom) in enumerate(
                    crop_boxes
                ):
                    frame_pixels = stack[top:bottom, left:right].copy()
                    if frame_pixels.shape != (height, width, 4):
                        raise _BatchGeometryError(
                            "Cropped frame dimensions do not match request: "
                            f"expected {(height, width, 4)}, "
                            f"got {frame_pixels.shape}"
                        )
                    if frame_sink is None:
                        frames.append(
                            frame_pixels.astype(np.float32) / 255.0
                        )
                    else:
                        frame_sink(start + chunk_index, frame_pixels)
            finally:
                await page.evaluate("resetSeekAndAppend()")
            start += current_chunk_size
    except _BatchGeometryError as error:
        print(f"{error}; falling back to sequential capture")
        return await _capture_png_frames_sequential_at_times(
            page,
            seek_times,
            dimensions,
            hide_grid,
            omit_background,
            frame_sink,
        )
    return frames


async def _capture_png_frames_at_times(
    page: Page,
    seek_times: Sequence[float],
    *,
    width: int | None = None,
    height: int | None = None,
    strategy: str = "sequential",
    hide_grid: bool = False,
    omit_background: bool = False,
    frame_sink: Callable[[int, np.ndarray], None] | None = None,
) -> list[np.ndarray]:
    """Capture ordered explicit times from an initialized animation page.

    ``width`` and ``height`` are paired logical CSS dimensions. Scale-1 output
    is enforced with Playwright's CSS screenshot scale. Batched capture requires
    explicit dimensions; sequential capture may retain the SVG's resolved size.
    """
    normalized_times = _normalize_seek_times(seek_times)
    dimensions = _normalize_raster_dimensions(width, height)
    normalized_strategy = _normalize_capture_strategy(strategy)
    if not isinstance(hide_grid, bool) or not isinstance(omit_background, bool):
        raise ValueError("hide_grid and omit_background must be booleans")
    if normalized_strategy == "batched" and dimensions is None:
        raise ValueError(
            "batched capture requires explicit width and height"
        )
    if not normalized_times:
        return []
    device_pixel_ratio = await page.evaluate("window.devicePixelRatio")
    if (
        not isinstance(device_pixel_ratio, (int, float))
        or abs(float(device_pixel_ratio) - 1.0) > 1e-6
    ):
        raise ValueError(
            "Stage 6C scale-1 capture requires devicePixelRatio == 1"
        )
    if normalized_strategy == "batched":
        assert dimensions is not None
        return await _capture_png_frames_batched_at_times(
            page,
            normalized_times,
            dimensions,
            hide_grid,
            omit_background,
            frame_sink,
        )
    return await _capture_png_frames_sequential_at_times(
        page,
        normalized_times,
        dimensions,
        hide_grid,
        omit_background,
        frame_sink,
    )


async def capture_png_frames_at_times(
    page: Page,
    seek_times: Sequence[float],
    *,
    width: int | None = None,
    height: int | None = None,
    strategy: str = "sequential",
    hide_grid: bool = False,
    omit_background: bool = False,
) -> list[np.ndarray]:
    """Return ordered RGBA frames for explicit times on an initialized page."""
    return await _capture_png_frames_at_times(
        page,
        seek_times,
        width=width,
        height=height,
        strategy=strategy,
        hide_grid=hide_grid,
        omit_background=omit_background,
    )
