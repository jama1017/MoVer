import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from mover.converter.mover_converter import (
    capture_frames_server_driven,
    capture_png_frames_at_times,
)
from mover.converter.raster_capture import (
    _BatchGeometryError,
    _capture_png_frames_sequential_at_times,
)


CONVERT_JS = (
    Path(__file__).parent.parent
    / "src"
    / "mover"
    / "converter"
    / "assets"
    / "convert.js"
)
GSAP_JS = CONVERT_JS.parent / "gsap.min.js"


class ConverterDomTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.playwright = await async_playwright().start()
        launch_options = {"headless": True}
        executable_path = os.environ.get("MOVER_BROWSER_TEST_EXECUTABLE")
        if executable_path:
            launch_options["executable_path"] = executable_path

        try:
            self.browser = await self.playwright.chromium.launch(**launch_options)
        except PlaywrightError as error:
            await self.playwright.stop()
            if os.environ.get("MOVER_REQUIRE_BROWSER") == "1":
                raise
            self.skipTest(f"Chromium is unavailable: {error}")

        self.page = await self.browser.new_page()
        await self.page.set_content(
            """<!doctype html>
            <style>
                html, body {
                    margin: 0;
                    padding-left: 5px;
                }
                body {
                    display: flex;
                }
                svg {
                    background-color: white;
                    background-image: linear-gradient(rgb(255, 0, 0), rgb(255, 0, 0));
                    background-repeat: no-repeat;
                }
                #shape {
                    opacity: 0.5;
                }
                body > svg #shape {
                    stroke-width: 2px;
                }
            </style>
            <body style="padding: 3px 4px !important; background: white">
                <p id="prompt" style="display: inline-block !important; color: blue">Prompt</p>
                <svg id="source" width="20" height="20" style="display: inline">
                    <style>
                        #shape { stroke: black; }
                        #paint stop { stop-opacity: 0.5; }
                    </style>
                    <defs>
                        <linearGradient id="paint">
                            <stop stop-color="green"/>
                        </linearGradient>
                    </defs>
                    <circle id="shape" cx="10" cy="10" r="3" fill='url("#paint")'/>
                </svg>
                <br id="break" style="display: block">
                <div id="existing" style="display: flex !important">Keep me</div>
                <script>
                    window.tl_to_use = {
                        seekCalls: [],
                        currentTotalTime: 1,
                        isPaused: true,
                        seek(time) {
                            this.seekCalls.push(time);
                            this.renderedTime = time;
                            this.currentTotalTime = time;
                            return this;
                        },
                        pause() {
                            this.isPaused = true;
                            return this;
                        },
                        play() {
                            this.isPaused = false;
                            return this;
                        },
                        paused(value) {
                            if (arguments.length === 0) {
                                return this.isPaused;
                            }
                            this.isPaused = value;
                            return this;
                        },
                        totalTime(value) {
                            if (arguments.length === 0) {
                                return this.currentTotalTime;
                            }
                            this.currentTotalTime = value;
                            this.renderedTime = value;
                            return this;
                        },
                        totalDuration() { return 1; },
                        getChildren() { return []; },
                        getTweensOf() { return []; },
                        totalProgress() {},
                    };
                    window.getAnimationInfo = fps => ({
                        animDuration: 1,
                        fps,
                        steps: 1,
                    });
                    window.seekToFrame = frameIndex => {
                        document.querySelector("#shape")
                            .setAttribute("cx", String(10 + frameIndex));
                    };
                </script>
            </body>"""
        )
        await self.page.add_script_tag(path=str(CONVERT_JS))
        self.initial_state = await self._snapshot_original_state()

    async def asyncTearDown(self) -> None:
        if hasattr(self, "browser"):
            await self.browser.close()
        if hasattr(self, "playwright"):
            await self.playwright.stop()

    async def _snapshot_original_state(self) -> dict:
        return await self.page.evaluate(
            """() => ({
                rootStyle: document.documentElement.getAttribute("style") || null,
                bodyStyle: document.body.getAttribute("style"),
                elements: Array.from(document.body.children)
                    .filter(element => element.tagName !== "SCRIPT")
                    .map(element => ({
                        id: element.id,
                        style: element.getAttribute("style"),
                    })),
            })"""
        )

    async def _assert_reset_state(self, expected_state: dict | None = None) -> None:
        self.assertEqual(
            await self._snapshot_original_state(),
            expected_state or self.initial_state,
        )
        self.assertEqual(
            await self.page.locator("[data-mover-batch-frame]").count(),
            0,
        )
        self.assertEqual(
            await self.page.locator("[data-mover-batch-container]").count(),
            0,
        )

    async def test_frame_sizes_grid_control_ids_and_reset(self) -> None:
        await self.page.set_viewport_size({"width": 128, "height": 256})
        expected_default_state = await self._snapshot_original_state()
        count = await self.page.evaluate(
            "seekAndAppendToDomUsingTimes([0, 0.5])"
        )
        self.assertEqual(count, 2)

        default_state = await self.page.evaluate(
            """() => Array.from(
                document.querySelectorAll("[data-mover-batch-frame]")
            ).map(svg => {
                const rect = svg.getBoundingClientRect();
                return {
                    rect: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                    },
                    svgCount: svg.tagName === "svg" ? 1 : 0,
                    backgroundImage: getComputedStyle(svg).backgroundImage,
                    paintId: svg.querySelector("linearGradient").id,
                    shapeId: svg.querySelector("circle").id,
                    fill: svg.querySelector("circle").getAttribute("fill"),
                    strokeWidth: getComputedStyle(
                        svg.querySelector("circle")
                    ).strokeWidth,
                    styleText: svg.querySelector("style").textContent,
                };
            })"""
        )
        self.assertEqual(
            [state["rect"] for state in default_state],
            [
                {"x": 0, "y": 0, "width": 128, "height": 128},
                {"x": 0, "y": 128, "width": 128, "height": 128},
            ],
        )
        self.assertEqual([state["svgCount"] for state in default_state], [1, 1])
        self.assertTrue(
            all("linear-gradient" in state["backgroundImage"] for state in default_state)
        )
        self.assertEqual(
            [state["paintId"] for state in default_state],
            ["mover_frame_0_paint", "mover_frame_1_paint"],
        )
        self.assertEqual(
            [state["shapeId"] for state in default_state],
            ["shape", "shape"],
        )
        self.assertEqual(
            [state["strokeWidth"] for state in default_state],
            ["2px", "2px"],
        )
        self.assertIn("#mover_frame_0_paint", default_state[0]["fill"])
        self.assertIn("#mover_frame_1_paint", default_state[1]["fill"])
        self.assertIn(
            "#mover_frame_0_paint stop",
            default_state[0]["styleText"],
        )
        self.assertIn(
            "#mover_frame_1_paint stop",
            default_state[1]["styleText"],
        )

        screenshot = Image.open(
            io.BytesIO(await self.page.screenshot(type="png", full_page=True))
        )
        self.assertEqual(screenshot.size, (128, 256))
        self.assertEqual(screenshot.convert("RGB").getpixel((1, 1)), (255, 0, 0))

        self.assertTrue(await self.page.evaluate("resetSeekAndAppend()"))
        await self._assert_reset_state(expected_default_state)
        restored_gsap_state = await self.page.evaluate(
            """() => ({
                targetTime: tl_to_use.totalTime(),
                targetPaused: tl_to_use.paused(),
            })"""
        )
        self.assertEqual(
            restored_gsap_state,
            {
                "targetTime": 1,
                "targetPaused": True,
            },
        )

        await self.page.set_viewport_size({"width": 512, "height": 1024})
        expected_large_state = await self._snapshot_original_state()
        count = await self.page.evaluate(
            "seekAndAppendToDomUsingTimes([0.5, 0.5], 512, true)"
        )
        self.assertEqual(count, 2)
        large_state = await self.page.evaluate(
            """() => Array.from(
                document.querySelectorAll("[data-mover-batch-frame]")
            ).map(svg => {
                const rect = svg.getBoundingClientRect();
                return {
                    rect: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                    },
                    backgroundImage: getComputedStyle(svg).backgroundImage,
                    rootId: svg.id,
                };
            })"""
        )
        self.assertEqual(
            [state["rect"] for state in large_state],
            [
                {"x": 0, "y": 0, "width": 512, "height": 512},
                {"x": 0, "y": 512, "width": 512, "height": 512},
            ],
        )
        self.assertEqual(
            [state["backgroundImage"] for state in large_state],
            ["none", "none"],
        )
        self.assertEqual(
            [state["rootId"] for state in large_state],
            ["source", "source"],
        )

        screenshot = Image.open(
            io.BytesIO(await self.page.screenshot(type="png", full_page=True))
        )
        self.assertEqual(screenshot.size, (512, 1024))
        self.assertEqual(screenshot.convert("RGB").getpixel((1, 1)), (255, 255, 255))

        self.assertTrue(await self.page.evaluate("resetSeekAndAppend()"))
        await self._assert_reset_state(expected_large_state)

    async def test_rectangular_batched_capture_matches_sequential_times(
        self,
    ) -> None:
        expected_viewport = {"width": 160, "height": 128}
        await self.page.set_viewport_size(expected_viewport)
        await self.page.evaluate(
            """() => {
                const source = document.querySelector("#source");
                const shape = document.querySelector("#shape");
                source.setAttribute("viewBox", "0 0 20 20");
                source.setAttribute("preserveAspectRatio", "xMidYMid meet");
                const originalSeek = tl_to_use.seek.bind(tl_to_use);
                tl_to_use.seek = time => {
                    originalSeek(time);
                    shape.setAttribute("cx", String(6 + 8 * time));
                    return tl_to_use;
                };
            }"""
        )
        expected_state = await self._snapshot_original_state()
        times = [0.0, 0.25, 0.5, 0.5, 1.0]
        sequential = await capture_png_frames_at_times(
            self.page,
            times,
            width=80,
            height=40,
            strategy="sequential",
            hide_grid=True,
        )
        with patch(
            "mover.converter.raster_capture."
            "_MAX_BATCH_FRAMES_PER_SCREENSHOT",
            3,
        ), patch(
            "mover.converter.raster_capture."
            "_capture_png_frames_sequential_at_times",
            side_effect=AssertionError("unexpected sequential fallback"),
        ), patch.object(
            self.page,
            "set_viewport_size",
            side_effect=AssertionError("batch capture changed viewport"),
        ):
            batched = await capture_png_frames_at_times(
                self.page,
                times,
                width=80,
                height=40,
                strategy="batched",
                hide_grid=True,
            )

        self.assertEqual(len(sequential), len(times))
        self.assertEqual(len(batched), len(times))
        for frame_index, (sequential_frame, batched_frame) in enumerate(
            zip(sequential, batched)
        ):
            self.assertEqual(sequential_frame.shape, (40, 80, 4))
            self.assertEqual(batched_frame.shape, (40, 80, 4))
            np.testing.assert_allclose(
                batched_frame,
                sequential_frame,
                atol=1.5 / 255.0,
                err_msg=f"frame {frame_index}",
            )
        np.testing.assert_array_equal(batched[2], batched[3])
        self.assertEqual(self.page.viewport_size, expected_viewport)
        await self._assert_reset_state(expected_state)

    async def test_transparent_background_matches_between_strategies(
        self,
    ) -> None:
        await self.page.evaluate(
            """() => {
                document.body.style.setProperty(
                    "background",
                    "transparent",
                    "important"
                );
                const source = document.querySelector("#source");
                source.style.setProperty(
                    "background-color",
                    "transparent",
                    "important"
                );
                source.style.setProperty(
                    "background-image",
                    "none",
                    "important"
                );
            }"""
        )
        sequential = await capture_png_frames_at_times(
            self.page,
            [0.0, 0.0],
            width=80,
            height=40,
            strategy="sequential",
            hide_grid=True,
            omit_background=True,
        )
        with patch(
            "mover.converter.raster_capture."
            "_capture_png_frames_sequential_at_times",
            side_effect=AssertionError("unexpected sequential fallback"),
        ):
            batched = await capture_png_frames_at_times(
                self.page,
                [0.0, 0.0],
                width=80,
                height=40,
                strategy="batched",
                hide_grid=True,
                omit_background=True,
            )
        for frames in (sequential, batched):
            self.assertEqual(frames[0][0, 0, 3], 0.0)
            self.assertEqual(frames[0][-1, -1, 3], 0.0)

    async def test_viewbox_only_svg_batches_at_exact_dimensions(self) -> None:
        await self.page.evaluate(
            """() => {
                const source = document.querySelector("#source");
                source.removeAttribute("width");
                source.removeAttribute("height");
                source.setAttribute("viewBox", "0 0 320 180");
                source.classList.add("sized-source");
                const style = document.createElement("style");
                style.textContent = `
                    body > svg.sized-source {
                        border: 2px solid purple;
                        padding: 1px;
                    }
                `;
                document.head.appendChild(style);
            }"""
        )
        captures = {
            "sequential": await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=160,
                height=90,
                strategy="sequential",
                hide_grid=True,
            )
        }
        with patch(
            "mover.converter.raster_capture."
            "_capture_png_frames_sequential_at_times",
            side_effect=AssertionError("unexpected sequential fallback"),
        ):
            captures["batched"] = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=160,
                height=90,
                strategy="batched",
                hide_grid=True,
            )
        for strategy in ("sequential", "batched"):
            self.assertTrue(
                all(
                    frame.shape == (90, 160, 4)
                    for frame in captures[strategy]
                )
            )
        for sequential, batched in zip(
            captures["sequential"],
            captures["batched"],
        ):
            np.testing.assert_allclose(
                batched,
                sequential,
                atol=1.5 / 255.0,
            )

    async def test_nonuniform_page_background_falls_back_before_batching(
        self,
    ) -> None:
        await self.page.evaluate(
            """() => {
                document.body.style.setProperty(
                    "background-image",
                    "linear-gradient(red, blue)",
                    "important"
                );
                window.originalBatchHelper = seekAndAppendToDomUsingTimes;
                window.seekAndAppendToDomUsingTimes = () => {
                    throw new Error("batch helper should not run");
                };
            }"""
        )
        try:
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
                strategy="batched",
                hide_grid=True,
            )
        finally:
            await self.page.evaluate(
                """() => {
                    window.seekAndAppendToDomUsingTimes =
                        window.originalBatchHelper;
                }"""
            )
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(frame.shape == (40, 80, 4) for frame in frames))

    async def test_animated_page_background_uses_sequential_fallback(
        self,
    ) -> None:
        await self.page.evaluate(
            """() => {
                const originalGetTweensOf = tl_to_use.getTweensOf.bind(
                    tl_to_use
                );
                window.originalGetTweensOf = originalGetTweensOf;
                tl_to_use.getTweensOf = element => (
                    element === document.body ? [{}] : originalGetTweensOf(element)
                );
                window.originalBatchHelper = seekAndAppendToDomUsingTimes;
                window.seekAndAppendToDomUsingTimes = () => {
                    throw new Error("batch helper should not run");
                };
            }"""
        )
        try:
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
                strategy="batched",
                hide_grid=True,
            )
        finally:
            await self.page.evaluate(
                """() => {
                    tl_to_use.getTweensOf = window.originalGetTweensOf;
                    window.seekAndAppendToDomUsingTimes =
                        window.originalBatchHelper;
                }"""
            )
        self.assertEqual(len(frames), 2)

    async def test_callback_page_background_uses_sequential_fallback(
        self,
    ) -> None:
        await self.page.evaluate(
            """() => {
                const source = document.querySelector("#source");
                source.style.setProperty(
                    "background-color",
                    "transparent",
                    "important"
                );
                source.style.setProperty(
                    "background-image",
                    "none",
                    "important"
                );
                const originalSeek = tl_to_use.seek.bind(tl_to_use);
                tl_to_use.seek = time => {
                    originalSeek(time);
                    document.body.style.setProperty(
                        "background-color",
                        time < 0.5 ? "rgb(255, 0, 0)" : "rgb(0, 0, 255)",
                        "important"
                    );
                    return tl_to_use;
                };
            }"""
        )
        with patch(
            "mover.converter.raster_capture."
            "_capture_png_frames_sequential_at_times",
            wraps=_capture_png_frames_sequential_at_times,
        ) as sequential_capture:
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
                strategy="batched",
                hide_grid=True,
            )
        self.assertEqual(len(frames), 2)
        self.assertEqual(sequential_capture.await_count, 1)

    async def test_root_svg_compositing_preserves_static_background(
        self,
    ) -> None:
        await self.page.evaluate(
            """() => {
                document.body.style.setProperty(
                    "background-color",
                    "red",
                    "important"
                );
                document.querySelector("#source").style.setProperty(
                    "opacity",
                    "0.5",
                    "important"
                );
            }"""
        )
        sequential = await capture_png_frames_at_times(
            self.page,
            [0.0, 1.0],
            width=80,
            height=40,
            strategy="sequential",
            hide_grid=True,
        )
        with patch(
            "mover.converter.raster_capture."
            "_capture_png_frames_sequential_at_times",
            side_effect=AssertionError("unexpected sequential fallback"),
        ):
            batched = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
                strategy="batched",
                hide_grid=True,
            )
        for sequential_frame, batched_frame in zip(sequential, batched):
            for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                np.testing.assert_allclose(
                    batched_frame[y, x],
                    sequential_frame[y, x],
                    atol=1.5 / 255.0,
                )

    async def test_callback_gradient_triggers_full_sequential_fallback(
        self,
    ) -> None:
        await self.page.evaluate(
            """() => {
                const originalSeek = tl_to_use.seek.bind(tl_to_use);
                tl_to_use.seek = time => {
                    originalSeek(time);
                    document.body.style.setProperty(
                        "background-image",
                        time < 0.5
                            ? "none"
                            : "linear-gradient(red, blue)",
                        "important"
                    );
                    return tl_to_use;
                };
            }"""
        )
        with patch(
            "mover.converter.raster_capture."
            "_capture_png_frames_sequential_at_times",
            wraps=_capture_png_frames_sequential_at_times,
        ) as sequential_capture:
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
                strategy="batched",
                hide_grid=True,
            )
        self.assertEqual(len(frames), 2)
        self.assertEqual(sequential_capture.await_count, 1)

    async def test_batch_screenshot_failure_resets_before_fallback(
        self,
    ) -> None:
        original_screenshot = self.page.screenshot
        call_count = 0

        async def fail_first_screenshot(**options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PlaywrightError("injected batch screenshot failure")
            return await original_screenshot(**options)

        with patch.object(
            self.page,
            "screenshot",
            side_effect=fail_first_screenshot,
        ):
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
                strategy="batched",
                hide_grid=True,
            )
        self.assertEqual(len(frames), 2)
        await self._assert_reset_state()

    async def test_invalid_batch_geometry_resets_before_fallback(
        self,
    ) -> None:
        with patch(
            "mover.converter.raster_capture._validate_batch_geometry",
            side_effect=_BatchGeometryError("injected geometry failure"),
        ), patch(
            "mover.converter.raster_capture."
            "_capture_png_frames_sequential_at_times",
            wraps=_capture_png_frames_sequential_at_times,
        ) as sequential_capture:
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
                strategy="batched",
                hide_grid=True,
            )
        self.assertEqual(len(frames), 2)
        self.assertEqual(sequential_capture.await_count, 1)
        await self._assert_reset_state()

    async def test_nested_source_svg_uses_sequential_fallback(self) -> None:
        await self.page.evaluate(
            """() => {
                const wrapper = document.createElement("section");
                const source = document.querySelector("#source");
                source.replaceWith(wrapper);
                wrapper.appendChild(source);
            }"""
        )
        frames = await capture_png_frames_at_times(
            self.page,
            [0.0, 1.0],
            width=80,
            height=40,
            strategy="batched",
            hide_grid=True,
        )
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(frame.shape == (40, 80, 4) for frame in frames))

    async def test_fewer_than_two_frames_fit_uses_full_sequential_fallback(
        self,
    ) -> None:
        with patch(
            "mover.converter.raster_capture."
            "_capture_png_frames_sequential_at_times",
            wraps=_capture_png_frames_sequential_at_times,
        ) as sequential_capture:
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=700,
                height=700,
                strategy="batched",
                hide_grid=True,
            )
        self.assertEqual(len(frames), 2)
        self.assertEqual(sequential_capture.await_count, 1)
        self.assertTrue(all(frame.shape == (700, 700, 4) for frame in frames))

    async def test_scale_one_capture_rejects_nonunit_dpr(self) -> None:
        page = await self.browser.new_page(device_scale_factor=2)
        try:
            with self.assertRaisesRegex(ValueError, "devicePixelRatio == 1"):
                await capture_png_frames_at_times(page, [0.0])
        finally:
            await page.close()

    async def test_batch_capture_restores_scroll_position(self) -> None:
        await self.page.set_viewport_size({"width": 1280, "height": 200})
        await self.page.evaluate(
            """() => {
                document.body.style.setProperty(
                    "display",
                    "block",
                    "important"
                );
                const spacer = document.createElement("div");
                spacer.id = "scroll-spacer";
                spacer.style.height = "2000px";
                document.body.appendChild(spacer);
                window.scrollTo(0, 500);
            }"""
        )
        before = await self.page.evaluate("window.scrollY")
        await capture_png_frames_at_times(
            self.page,
            [0.0, 1.0],
            width=128,
            height=128,
            strategy="batched",
            hide_grid=True,
        )
        after = await self.page.evaluate("window.scrollY")
        self.assertEqual(before, 500)
        self.assertEqual(after, before)

    async def test_automatic_batch_schedule_includes_endpoint_and_static_frame(
        self,
    ) -> None:
        await self.page.evaluate(
            """() => {
                window.getAnimationInfo = () => ({
                    animDuration: 1,
                    fps: 2,
                    steps: 2,
                });
                tl_to_use.seekCalls = [];
            }"""
        )
        self.assertEqual(await self.page.evaluate("seekAndAppendToDom()"), 3)
        self.assertEqual(
            await self.page.evaluate("tl_to_use.seekCalls"),
            [0, 0.5, 1],
        )
        self.assertTrue(await self.page.evaluate("resetSeekAndAppend()"))
        await self._assert_reset_state()

        await self.page.evaluate(
            """() => {
                window.getAnimationInfo = () => ({
                    animDuration: 0,
                    fps: 60,
                    steps: 0,
                });
                tl_to_use.seekCalls = [];
            }"""
        )
        self.assertEqual(await self.page.evaluate("seekAndAppendToDom()"), 1)
        self.assertEqual(
            await self.page.evaluate("tl_to_use.seekCalls"),
            [0],
        )
        self.assertTrue(await self.page.evaluate("resetSeekAndAppend()"))
        await self._assert_reset_state()

    async def test_scene_size_falls_back_to_viewbox_and_preserves_paint_fill(
        self,
    ) -> None:
        await self.page.add_script_tag(path=str(GSAP_JS))
        result = await self.page.evaluate(
            """() => {
                const source = document.querySelector("#source");
                const shape = document.querySelector("#shape");
                source.removeAttribute("width");
                source.removeAttribute("height");
                source.setAttribute("viewBox", "0 0 320 180");
                shape.setAttribute("fill", "url(#paint)");
                const viewBoxSceneSize = getSvgSceneSize(source);

                const objectWithPaintServer = createObjectList([shape], [])[0];
                const keyframes = convertToKeyframes([shape]);
                const transformations = getAllTransformationValues([shape], 1);
                const rendered = createRenderedData(
                    [shape],
                    { spatial: {} },
                    { spatial: [], visual: [], svgAttributes: [] },
                    1
                );

                shape.setAttribute("fill", "#ffffff");
                const objectWithHexFill = createObjectList([shape], [])[0];
                source.setAttribute("width", "640px");
                source.setAttribute("height", "480px");
                const unitSceneSize = getSvgSceneSize(source);
                source.removeAttribute("width");
                source.removeAttribute("height");
                source.removeAttribute("viewBox");
                const missingSceneSize = getSvgSceneSize(source);
                return {
                    directSceneSize: viewBoxSceneSize,
                    keyframeSceneSize: keyframes.info["scene-size"],
                    transformationSceneSize:
                        transformations.info["scene-size"],
                    renderedSceneSize: rendered.info["scene-size"],
                    paintServerFill: objectWithPaintServer.fill,
                    hexFill: objectWithHexFill.fill,
                    unitSceneSize,
                    missingSceneSize,
                };
            }"""
        )
        expected_size = {"width": 320, "height": 180}
        self.assertEqual(result["directSceneSize"], expected_size)
        self.assertEqual(result["keyframeSceneSize"], expected_size)
        self.assertEqual(result["transformationSceneSize"], expected_size)
        self.assertEqual(result["renderedSceneSize"], expected_size)
        self.assertEqual(result["paintServerFill"], "url(#paint)")
        self.assertEqual(result["hexFill"], "white")
        self.assertEqual(
            result["unitSceneSize"],
            {"width": 640, "height": 480},
        )
        self.assertEqual(
            result["missingSceneSize"],
            {"width": None, "height": None},
        )

    async def test_invalid_options_and_seek_failure_do_not_mutate_page(self) -> None:
        for expression in (
            "seekAndAppendToDomUsingTimes([0], 0)",
            "seekAndAppendToDomUsingTimes([0], 128, 'yes')",
            "seekAndAppendToDomUsingTimes([NaN])",
        ):
            with self.assertRaises(Exception):
                await self.page.evaluate(expression)
            await self._assert_reset_state()

        await self.page.evaluate(
            """() => {
                window.tl_to_use.seek = time => {
                    if (time > 0) throw new Error("injected failure");
                };
            }"""
        )
        expected_failure_state = await self._snapshot_original_state()
        with self.assertRaisesRegex(Exception, "injected failure"):
            await self.page.evaluate("seekAndAppendToDomUsingTimes([0, 1])")
        await self._assert_reset_state(expected_failure_state)
        self.assertFalse(await self.page.evaluate("resetSeekAndAppend()"))

    async def test_real_gsap_selected_timeline_renders_and_restores(self) -> None:
        await self.page.add_script_tag(path=str(GSAP_JS))
        initial_state = await self.page.evaluate(
            """() => {
                const shape = document.querySelector("#shape");
                window.targetObject = { value: 0 };
                window.tl_to_use = gsap.timeline({ paused: true });
                tl_to_use.to(
                    targetObject,
                    { value: 100, duration: 1, ease: "none" }
                );
                window.targetUpdateCount = 0;
                tl_to_use.eventCallback(
                    "onUpdate",
                    () => {
                        targetUpdateCount++;
                        shape.setAttribute(
                            "data-progress",
                            String(Math.round(targetObject.value))
                        );
                    }
                );
                tl_to_use.timeScale(2);
                return {
                    targetTime: tl_to_use.totalTime(),
                    targetPaused: tl_to_use.paused(),
                };
            }"""
        )

        await self.page.evaluate("seekToTime(0.5)")
        synchronized = await self.page.evaluate(
            """() => ({
                targetValue: targetObject.value,
                targetTime: tl_to_use.totalTime(),
                progress: document.querySelector("#shape")
                    .getAttribute("data-progress"),
            })"""
        )
        self.assertAlmostEqual(synchronized["targetValue"], 50, places=4)
        self.assertAlmostEqual(synchronized["targetTime"], 0.5, places=6)
        self.assertEqual(synchronized["progress"], "50")

        before_batch = await self.page.evaluate(
            """async () => {
                tl_to_use.reversed(true);
                tl_to_use.paused(true);
                const state = {
                    targetTime: tl_to_use.totalTime(),
                    targetPaused: tl_to_use.paused(),
                    targetReversed: tl_to_use.reversed(),
                };
                await seekAndAppendToDomUsingTimes([0, 0.75], 128, false);
                return state;
            }"""
        )
        update_count_before_reset = await self.page.evaluate(
            "targetUpdateCount"
        )
        self.assertEqual(
            await self.page.locator(
                "body > svg:not([data-mover-batch-frame]) #shape"
            ).get_attribute("data-progress"),
            "75",
        )
        self.assertTrue(await self.page.evaluate("resetSeekAndAppend()"))
        self.assertGreater(
            await self.page.evaluate("targetUpdateCount"),
            update_count_before_reset,
        )
        after_batch = await self.page.evaluate(
            """() => ({
                targetTime: tl_to_use.totalTime(),
                targetPaused: tl_to_use.paused(),
                targetReversed: tl_to_use.reversed(),
                progress: document.querySelector("#shape")
                    .getAttribute("data-progress"),
            })"""
        )
        self.assertAlmostEqual(
            after_batch["targetTime"],
            before_batch["targetTime"],
            places=6,
        )
        self.assertEqual(
            after_batch["targetPaused"],
            before_batch["targetPaused"],
        )
        self.assertEqual(
            after_batch["targetReversed"],
            before_batch["targetReversed"],
        )
        self.assertEqual(after_batch["progress"], "50")
        self.assertTrue(initial_state["targetPaused"])

    async def test_batch_clone_keeps_endpoint_svg_transform_visible(
        self,
    ) -> None:
        await self.page.add_script_tag(path=str(GSAP_JS))
        clone_state = await self.page.evaluate(
            """async () => {
                const source = document.querySelector("#source");
                const shape = document.querySelector("#shape");
                source.setAttribute("width", "80");
                source.setAttribute("height", "40");
                source.setAttribute("viewBox", "0 0 80 40");
                shape.setAttribute("cx", "10");
                shape.setAttribute("cy", "20");
                shape.setAttribute("r", "6");
                shape.setAttribute(
                    "data-layout-width",
                    String(source.getBoundingClientRect().width)
                );
                window.tl_to_use = gsap.timeline({paused: true});
                tl_to_use.to(shape, {
                    x: 50,
                    duration: 1,
                    ease: "none",
                    onUpdate: () => {
                        shape.setAttribute(
                            "data-layout-width",
                            String(source.getBoundingClientRect().width)
                        );
                    },
                });
                await seekAndAppendToDomUsingTimes(
                    [0, 0.5, 1],
                    80,
                    40,
                    true
                );
                return Array.from(
                    document.querySelectorAll("[data-mover-batch-frame]")
                ).map(wrapper => {
                    const clone = wrapper.querySelector("#shape");
                    const wrapperRect = wrapper.getBoundingClientRect();
                    const cloneRect = clone.getBoundingClientRect();
                    return {
                        style: clone.getAttribute("style"),
                        x: cloneRect.x - wrapperRect.x,
                        y: cloneRect.y - wrapperRect.y,
                        width: cloneRect.width,
                        height: cloneRect.height,
                        layoutWidth: clone.getAttribute(
                            "data-layout-width"
                        ),
                    };
                });
            }"""
        )
        try:
            self.assertEqual(len(clone_state), 3)
            self.assertAlmostEqual(clone_state[0]["x"], 4, delta=1)
            self.assertAlmostEqual(clone_state[1]["x"], 29, delta=1)
            self.assertAlmostEqual(clone_state[2]["x"], 54, delta=1)
            self.assertGreater(clone_state[2]["width"], 10)
            self.assertEqual(
                [state["layoutWidth"] for state in clone_state],
                ["80", "80", "80"],
            )
        finally:
            await self.page.evaluate("resetSeekAndAppend()")

    async def test_rendered_data_runs_callback_derived_svg_updates(self) -> None:
        await self.page.add_script_tag(path=str(GSAP_JS))
        rendered = await self.page.evaluate(
            """() => {
                const shape = document.querySelector("#shape");
                const proxy = { value: 0 };
                let updateCount = 0;
                window.tl_to_use = gsap.timeline({ paused: true });
                tl_to_use.to(proxy, {
                    value: 100,
                    duration: 1,
                    ease: "none",
                    onUpdate: () => {
                        updateCount++;
                        shape.setAttribute(
                            "data-progress",
                            String(Math.round(proxy.value))
                        );
                    },
                });
                tl_to_use.totalProgress(1);

                const data = createRenderedData(
                    [shape],
                    { spatial: {} },
                    {
                        spatial: [],
                        visual: [],
                        svgAttributes: ["data-progress"],
                    },
                    2
                );
                return {
                    samples: data.shape["data-progress"],
                    finalAttribute: shape.getAttribute("data-progress"),
                    updateCount,
                };
            }"""
        )
        self.assertEqual(rendered["samples"], ["0", "50", "100"])
        self.assertEqual(rendered["finalAttribute"], "0")
        self.assertGreaterEqual(rendered["updateCount"], 4)

    async def test_server_capture_restores_page_state_after_success_and_failure(
        self,
    ) -> None:
        await self.page.add_script_tag(path=str(GSAP_JS))
        await self.page.evaluate(
            """() => {
                const shape = document.querySelector("#shape");
                window.captureProxy = { value: 0 };
                window.tl_to_use = gsap.timeline({ paused: true });
                tl_to_use.to(captureProxy, {
                    value: 100,
                    duration: 1,
                    ease: "none",
                    onUpdate: () => {
                        shape.setAttribute(
                            "data-progress",
                            String(Math.round(captureProxy.value))
                        );
                    },
                });
                tl_to_use.totalTime(0.25, false).pause();

                const devtools = document.createElement("div");
                devtools.id = "GSDevTools";
                devtools.setAttribute(
                    "style",
                    "display: flex !important; color: red"
                );
                document.body.appendChild(devtools);

                const controls = document.createElement("div");
                controls.className = "gs-dev-tools-controls";
                controls.setAttribute(
                    "style",
                    "display: grid !important; color: blue"
                );
                document.body.appendChild(controls);
            }"""
        )

        snapshot_js = """() => ({
            timelineTime: tl_to_use.totalTime(),
            timelinePaused: tl_to_use.paused(),
            progress: document.querySelector("#shape")
                .getAttribute("data-progress"),
            devtoolsStyle: document.querySelector("#GSDevTools")
                .getAttribute("style"),
            controlsStyle: document.querySelector(".gs-dev-tools-controls")
                .getAttribute("style"),
        })"""
        initial_state = await self.page.evaluate(snapshot_js)

        with tempfile.TemporaryDirectory() as temp_dir:
            frames, duration = await capture_frames_server_driven(
                self.page,
                str(Path(temp_dir) / "unused"),
                fps=2,
                output_format="svg",
                in_memory=True,
            )
        self.assertEqual(duration, 1)
        self.assertEqual(len(frames), 3)
        self.assertEqual(await self.page.evaluate(snapshot_js), initial_state)

        await self.page.evaluate(
            """() => {
                tl_to_use.totalTime(0.4, false).pause();
                window.originalSeekToFrame = window.seekToFrame;
                window.seekToFrame = () => {
                    throw new Error("injected server capture failure");
                };
            }"""
        )
        failure_state = await self.page.evaluate(snapshot_js)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                Exception,
                "injected server capture failure",
            ):
                await capture_frames_server_driven(
                    self.page,
                    str(Path(temp_dir) / "unused"),
                    fps=2,
                    output_format="svg",
                    in_memory=True,
                )
        self.assertEqual(await self.page.evaluate(snapshot_js), failure_state)
        await self.page.evaluate(
            "() => { window.seekToFrame = window.originalSeekToFrame; }"
        )

    async def test_real_browser_server_capture_and_grid_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            default_output = Path(temp_dir) / "default" / "frames"
            default_frames, default_duration = await capture_frames_server_driven(
                self.page,
                str(default_output),
                fps=1,
                output_format="png",
                in_memory=True,
            )
            hidden_output = Path(temp_dir) / "hidden" / "frames"
            hidden_frames, hidden_duration = await capture_frames_server_driven(
                self.page,
                str(hidden_output),
                fps=1,
                output_format="png",
                in_memory=True,
                hide_grid=True,
            )
            disk_png_output = Path(temp_dir) / "disk-png"
            await capture_frames_server_driven(
                self.page,
                str(disk_png_output),
                fps=1,
                output_format="png",
            )

            self.assertEqual(default_duration, 1)
            self.assertEqual(hidden_duration, 1)
            self.assertEqual(len(default_frames), 2)
            self.assertEqual(len(hidden_frames), 2)
            self.assertFalse(default_output.parent.exists())
            self.assertFalse(hidden_output.parent.exists())
            disk_png_paths = sorted(disk_png_output.glob("frame_*.png"))
            self.assertEqual(len(disk_png_paths), 2)
            for memory_frame, disk_path in zip(default_frames, disk_png_paths):
                disk_frame = (
                    np.asarray(Image.open(disk_path).convert("RGBA"), dtype=np.float32)
                    / 255.0
                )
                np.testing.assert_array_equal(memory_frame, disk_frame)

            batched_memory_output = (
                Path(temp_dir) / "missing-batched" / "frames"
            )
            batched_frames, batched_duration = (
                await capture_frames_server_driven(
                    self.page,
                    str(batched_memory_output),
                    fps=1,
                    output_format="png",
                    in_memory=True,
                    hide_grid=True,
                    capture_strategy="batched",
                    width=64,
                    height=32,
                )
            )
            batched_disk_output = Path(temp_dir) / "disk-batched"
            with patch(
                "mover.converter.mover_converter.Image.fromarray",
                wraps=Image.fromarray,
            ) as image_fromarray:
                await capture_frames_server_driven(
                    self.page,
                    str(batched_disk_output),
                    fps=1,
                    output_format="png",
                    hide_grid=True,
                    capture_strategy="batched",
                    width=64,
                    height=32,
                )
            self.assertTrue(image_fromarray.call_args_list)
            self.assertTrue(
                all(
                    call.args[0].dtype == np.uint8
                    for call in image_fromarray.call_args_list
                )
            )
            self.assertEqual(batched_duration, 1)
            self.assertEqual(len(batched_frames), 2)
            self.assertFalse(batched_memory_output.parent.exists())
            batched_disk_paths = sorted(
                batched_disk_output.glob("frame_*.png")
            )
            self.assertEqual(len(batched_disk_paths), 2)
            for memory_frame, disk_path in zip(
                batched_frames,
                batched_disk_paths,
            ):
                self.assertEqual(memory_frame.shape, (32, 64, 4))
                disk_frame = (
                    np.asarray(
                        Image.open(disk_path).convert("RGBA"),
                        dtype=np.float32,
                    )
                    / 255.0
                )
                np.testing.assert_array_equal(memory_frame, disk_frame)

            default_pixel = tuple(
                round(float(channel) * 255)
                for channel in default_frames[0][0, 0, :3]
            )
            hidden_pixel = tuple(
                round(float(channel) * 255)
                for channel in hidden_frames[0][0, 0, :3]
            )
            self.assertEqual(default_pixel, (255, 0, 0))
            self.assertEqual(hidden_pixel, (255, 255, 255))
            self.assertIn(
                "linear-gradient",
                await self.page.locator("body > svg").evaluate(
                    "svg => getComputedStyle(svg).backgroundImage"
                ),
            )

            disk_svg_output = Path(temp_dir) / "disk-svg"
            await capture_frames_server_driven(
                self.page,
                str(disk_svg_output),
                fps=1,
                output_format="svg",
            )
            memory_svg_output = Path(temp_dir) / "missing-svg" / "frames"
            memory_svg_frames, svg_duration = await capture_frames_server_driven(
                self.page,
                str(memory_svg_output),
                fps=1,
                output_format="svg",
                in_memory=True,
            )
            self.assertEqual(svg_duration, 1)
            self.assertFalse(memory_svg_output.parent.exists())
            disk_svg_paths = sorted(disk_svg_output.glob("frame_*.svg"))
            self.assertEqual(len(disk_svg_paths), 2)
            self.assertEqual(
                [frame.getvalue() for frame in memory_svg_frames],
                [path.read_text() for path in disk_svg_paths],
            )
if __name__ == "__main__":
    unittest.main()
