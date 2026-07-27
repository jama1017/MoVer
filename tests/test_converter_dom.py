import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from mover.converter.mover_converter import capture_frames_server_driven
from mover.converter.raster_capture import (
    BatchCaptureError,
    _capture_sequential,
    _decode_rgba,
    capture_png_frames_at_times,
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
PROPERTY_REGISTRY = json.loads(
    (CONVERT_JS.parent / "property_registry.json").read_text(
        encoding="utf-8"
    )
)


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
                    <circle id="shape" cx="10" cy="10" r="3"
                        fill='url("#paint")'
                        data-external="url(https://example.com/a.svg#paint)"/>
                </svg>
                <br id="break" style="display: block">
                <div id="existing" style="display: flex !important">Keep me</div>
                <script>
                    window.tl_to_use = {
                        seekCalls: [],
                        totalTimeCalls: [],
                        currentTotalTime: 1,
                        isPaused: true,
                        seek(time) {
                            this.seekCalls.push(time);
                            this.renderedTime = time;
                            this.currentTotalTime = time;
                            document.querySelector("#shape").setAttribute(
                                "cx",
                                String(7 + (6 * time)),
                            );
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
                        totalTime(value, suppressEvents) {
                            if (arguments.length === 0) {
                                return this.currentTotalTime;
                            }
                            if (suppressEvents === false) {
                                this.totalTimeCalls.push(value);
                            }
                            this.currentTotalTime = value;
                            this.renderedTime = value;
                            document.querySelector("#shape").setAttribute(
                                "cx",
                                String(7 + (6 * value)),
                            );
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
            await self.page.locator("body > [data-mover-batch-frame]").count(),
            0,
        )

    async def test_conversion_helpers_do_not_assign_page_globals(self) -> None:
        await self.page.add_script_tag(
            content="""
                const elem_data = "page-elem-data";
                const rect = "page-rect";
                const svgOffset = "page-svg-offset";
                const py_rect = "page-py-rect";
                const matrix = "page-matrix";
                const bb = "page-bb";
                const tpts = "page-tpts";
                const ndf2 = "page-ndf2";
                const ndf = "page-ndf";
            """
        )

        result = await self.page.evaluate(
            """() => {
                const shape = document.querySelector("#shape");
                shape.setAttribute("fill", "#ffffff");
                return {
                    aabb: getAABB(shape),
                    transformedAabb: getTransformedAABB(shape),
                    objectData: createObjectList([shape], [])[0],
                    sentinels: {
                        elem_data,
                        rect,
                        svgOffset,
                        py_rect,
                        matrix,
                        bb,
                        tpts,
                        ndf2,
                        ndf,
                    },
                };
            }"""
        )

        self.assertEqual(result["aabb"]["width"], 6)
        self.assertEqual(result["aabb"]["height"], 6)
        self.assertEqual(len(result["transformedAabb"]), 4)
        self.assertEqual(result["objectData"]["shape"], "circle")
        self.assertEqual(result["objectData"]["fill"], "white")
        self.assertEqual(
            result["sentinels"],
            {
                "elem_data": "page-elem-data",
                "rect": "page-rect",
                "svgOffset": "page-svg-offset",
                "py_rect": "page-py-rect",
                "matrix": "page-matrix",
                "bb": "page-bb",
                "tpts": "page-tpts",
                "ndf2": "page-ndf2",
                "ndf": "page-ndf",
            },
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
                document.querySelectorAll("body > [data-mover-batch-frame]")
            ).map(wrapper => {
                const rect = wrapper.getBoundingClientRect();
                const svg = wrapper.querySelector(":scope > svg");
                return {
                    rect: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                    },
                    svgCount: wrapper.querySelectorAll(":scope > svg").length,
                    backgroundImage: getComputedStyle(svg).backgroundImage,
                    paintId: svg.querySelector("linearGradient").id,
                    shapeId: svg.querySelector("circle").id,
                    fill: svg.querySelector("circle").getAttribute("fill"),
                    external: svg.querySelector("circle")
                        .getAttribute("data-external"),
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
        self.assertIn("#mover_frame_0_paint", default_state[0]["fill"])
        self.assertIn("#mover_frame_1_paint", default_state[1]["fill"])
        self.assertEqual(
            [state["external"] for state in default_state],
            ["url(https://example.com/a.svg#paint)"] * 2,
        )
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
                document.querySelectorAll("body > [data-mover-batch-frame]")
            ).map(wrapper => {
                const rect = wrapper.getBoundingClientRect();
                const svg = wrapper.querySelector(":scope > svg");
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
                tl_to_use.totalTimeCalls = [];
            }"""
        )
        self.assertEqual(await self.page.evaluate("seekAndAppendToDom()"), 3)
        self.assertEqual(
            await self.page.evaluate("tl_to_use.totalTimeCalls"),
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
                tl_to_use.totalTimeCalls = [];
            }"""
        )
        self.assertEqual(await self.page.evaluate("seekAndAppendToDom()"), 1)
        self.assertEqual(
            await self.page.evaluate("tl_to_use.totalTimeCalls"),
            [0],
        )
        self.assertTrue(await self.page.evaluate("resetSeekAndAppend()"))
        await self._assert_reset_state()

    async def test_fast_capture_matches_rectangular_sequential_pixels(self) -> None:
        await self.page.set_viewport_size({"width": 160, "height": 128})
        await self.page.evaluate(
            """() => {
                const defs = document.querySelector("#source defs");
                defs.insertAdjacentHTML(
                    "beforeend",
                    '<filter id="soft"><feGaussianBlur stdDeviation="0.2"/></filter>'
                    + '<mask id="visible"><rect width="100%" height="100%" fill="white"/></mask>'
                );
                const shape = document.querySelector("#shape");
                shape.setAttribute("filter", "url(#soft)");
                shape.setAttribute("mask", "url(#visible)");
                window._moverOriginalRaf = window.requestAnimationFrame;
                window._moverRafCalls = 0;
                window.requestAnimationFrame = callback => {
                    window._moverRafCalls += 1;
                    return window._moverOriginalRaf(callback);
                };
            }"""
        )
        expected_state = await self._snapshot_original_state()
        times = [0.0, 0.25, 0.5, 0.5, 1.0]
        try:
            with patch(
                "mover.converter.raster_capture._decode_rgba",
                wraps=_decode_rgba,
            ) as decode:
                batched = await capture_png_frames_at_times(
                    self.page,
                    times,
                    width=80,
                    height=40,
                    strategy="batched",
                    hide_grid=True,
                )
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(
                await self.page.evaluate("window._moverRafCalls"),
                2,
            )
        finally:
            await self.page.evaluate(
                """() => {
                    window.requestAnimationFrame = window._moverOriginalRaf;
                    delete window._moverOriginalRaf;
                    delete window._moverRafCalls;
                }"""
            )

        sequential = await capture_png_frames_at_times(
            self.page,
            times,
            width=80,
            height=40,
            strategy="sequential",
            hide_grid=True,
        )
        self.assertEqual(self.page.viewport_size, {"width": 160, "height": 128})
        self.assertEqual(len(batched), len(times))
        for index, (fast_frame, sequential_frame) in enumerate(
            zip(batched, sequential)
        ):
            self.assertEqual(fast_frame.shape, (40, 80, 4))
            np.testing.assert_allclose(
                fast_frame,
                sequential_frame,
                atol=1.5 / 255.0,
                err_msg=f"frame {index}",
            )
        np.testing.assert_array_equal(batched[2], batched[3])
        await self._assert_reset_state(expected_state)

    async def test_fast_capture_chunks_without_cross_frame_state(self) -> None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
        with patch(
            "mover.converter.raster_capture._plan_batch_chunk_size",
            return_value=2,
        ), patch(
            "mover.converter.raster_capture._decode_rgba",
            wraps=_decode_rgba,
        ) as decode:
            frames = await capture_png_frames_at_times(
                self.page,
                times,
                width=80,
                height=40,
            )
        self.assertEqual(len(frames), len(times))
        self.assertEqual(decode.call_count, 3)
        self.assertGreater(np.abs(frames[0] - frames[-1]).mean(), 0.001)
        await self._assert_reset_state()

    async def test_ineligible_nested_scene_reports_and_falls_back(self) -> None:
        await self.page.evaluate(
            """() => {
                const section = document.createElement("section");
                section.id = "scene-wrapper";
                const source = document.querySelector("#source");
                source.replaceWith(section);
                section.appendChild(source);
            }"""
        )
        expected_state = await self._snapshot_original_state()
        with self.assertLogs(
            "mover.converter.raster_capture",
            level="WARNING",
        ) as logs, patch(
            "mover.converter.raster_capture._capture_sequential",
            wraps=_capture_sequential,
        ) as sequential_capture:
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
            )
        self.assertEqual(sequential_capture.await_count, 1)
        self.assertIn("not a direct body child", "\n".join(logs.output))
        self.assertEqual([frame.shape for frame in frames], [(40, 80, 4)] * 2)
        await self._assert_reset_state(expected_state)

    async def test_fast_support_rejects_unsafe_page_state(self) -> None:
        results = await self.page.evaluate(
            """() => {
                const baseline = getBatchCaptureSupport();

                document.body.style.setProperty(
                    "background-image",
                    "linear-gradient(red, blue)",
                    "important"
                );
                const background = getBatchCaptureSupport();
                document.body.style.removeProperty("background-image");

                const source = document.querySelector("#source");
                source.style.setProperty(
                    "transform", "translate3d(0, 0, 0)", "important"
                );
                const identityTransform = getBatchCaptureSupport();
                source.style.setProperty(
                    "transform", "translateX(1px)", "important"
                );
                const transform = getBatchCaptureSupport();
                source.style.removeProperty("transform");

                const rootEffects = {};
                for (const [name, value] of [
                    ["filter", "blur(1px)"],
                    ["mix-blend-mode", "multiply"],
                    ["opacity", "0.9"],
                    ["clip-path", "inset(1px)"],
                    ["mask-image", "linear-gradient(black, transparent)"],
                ]) {
                    source.style.setProperty(name, value, "important");
                    rootEffects[name] = getBatchCaptureSupport();
                    source.style.removeProperty(name);
                }

                source.setAttribute("viewBox", "0 0 20 20");
                const letterboxing = getBatchCaptureSupport(80, 40);
                source.removeAttribute("viewBox");

                const text = document.createElementNS(
                    "http://www.w3.org/2000/svg", "text"
                );
                text.textContent = "unsafe";
                source.appendChild(text);
                const textContent = getBatchCaptureSupport();
                text.remove();

                const originalGetChildren = tl_to_use.getChildren;
                const detachedTarget = document.createElement("div");
                tl_to_use.getChildren = () => [{
                    targets: () => [detachedTarget],
                }];
                const detachedOutsideTarget = getBatchCaptureSupport();
                document.body.appendChild(detachedTarget);
                const connectedOutsideTarget = getBatchCaptureSupport();
                detachedTarget.remove();

                tl_to_use.getChildren = () => [{
                    targets: () => [document.body],
                }];
                const outsideTarget = getBatchCaptureSupport();
                tl_to_use.getChildren = originalGetChildren;
                return {
                    baseline,
                    background,
                    identityTransform,
                    transform,
                    rootEffects,
                    letterboxing,
                    textContent,
                    detachedOutsideTarget,
                    connectedOutsideTarget,
                    outsideTarget,
                };
            }"""
        )
        self.assertTrue(results["baseline"]["supported"])
        self.assertTrue(results["identityTransform"]["supported"])
        self.assertTrue(results["detachedOutsideTarget"]["supported"])
        for name in (
            "background",
            "transform",
            "letterboxing",
            "textContent",
            "connectedOutsideTarget",
            "outsideTarget",
        ):
            self.assertFalse(results[name]["supported"], name)
            self.assertTrue(results[name]["reason"])
        for name, result in results["rootEffects"].items():
            self.assertFalse(result["supported"], name)
            self.assertIn(name, result["reason"])

    async def test_scale_one_capture_rejects_nonunit_dpr(self) -> None:
        page = await self.browser.new_page(device_scale_factor=2)
        try:
            with self.assertRaisesRegex(ValueError, "devicePixelRatio == 1"):
                await capture_png_frames_at_times(
                    page,
                    [0.0],
                    width=80,
                    height=40,
                )
        finally:
            await page.close()

    async def test_geometry_failure_resets_before_sequential_fallback(self) -> None:
        with patch(
            "mover.converter.raster_capture._validate_batch_geometry",
            side_effect=BatchCaptureError("injected geometry failure"),
        ), patch(
            "mover.converter.raster_capture._capture_sequential",
            wraps=_capture_sequential,
        ) as sequential_capture:
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
            )
        self.assertEqual(sequential_capture.await_count, 1)
        self.assertEqual(len(frames), 2)
        await self._assert_reset_state()

    async def test_batch_screenshot_failure_resets_before_fallback(self) -> None:
        original_screenshot = self.page.screenshot
        screenshot_calls = 0

        async def fail_first_screenshot(**kwargs):
            nonlocal screenshot_calls
            screenshot_calls += 1
            if screenshot_calls == 1:
                raise PlaywrightError("injected batch screenshot failure")
            return await original_screenshot(**kwargs)

        with self.assertLogs(
            "mover.converter.raster_capture",
            level="WARNING",
        ), patch.object(
            self.page,
            "screenshot",
            side_effect=fail_first_screenshot,
        ):
            frames = await capture_png_frames_at_times(
                self.page,
                [0.0, 1.0],
                width=80,
                height=40,
            )
        self.assertEqual(len(frames), 2)
        await self._assert_reset_state()

    async def test_seek_failure_resets_and_propagates(self) -> None:
        await self.page.evaluate(
            """() => {
                window._moverOriginalSeekToTime = window.seekToTime;
                window.seekToTime = () => {
                    throw new Error("injected seek failure");
                };
            }"""
        )
        try:
            with patch(
                "mover.converter.raster_capture._capture_sequential",
                side_effect=AssertionError("seek failure must not fall back"),
            ), self.assertRaisesRegex(PlaywrightError, "injected seek failure"):
                await capture_png_frames_at_times(
                    self.page,
                    [0.0, 1.0],
                    width=80,
                    height=40,
                )
        finally:
            await self.page.evaluate(
                """() => {
                    window.seekToTime = window._moverOriginalSeekToTime;
                    delete window._moverOriginalSeekToTime;
                }"""
            )
        await self._assert_reset_state()

    async def test_viewbox_only_transparent_scene_keeps_exact_dimensions(
        self,
    ) -> None:
        await self.page.evaluate(
            """() => {
                document.documentElement.style.setProperty(
                    "background", "transparent", "important"
                );
                document.body.style.setProperty(
                    "background", "transparent", "important"
                );
                const source = document.querySelector("#source");
                source.removeAttribute("width");
                source.removeAttribute("height");
                source.setAttribute("viewBox", "0 0 20 20");
                source.style.setProperty(
                    "background", "transparent", "important"
                );
            }"""
        )
        expected_state = await self._snapshot_original_state()
        times = [0.0, 1.0]
        sequential = await capture_png_frames_at_times(
            self.page,
            times,
            width=160,
            height=90,
            strategy="sequential",
            hide_grid=True,
            omit_background=True,
        )
        batched = await capture_png_frames_at_times(
            self.page,
            times,
            width=160,
            height=90,
            hide_grid=True,
            omit_background=True,
        )
        for fast_frame, sequential_frame in zip(batched, sequential):
            self.assertEqual(fast_frame.shape, (90, 160, 4))
            np.testing.assert_allclose(
                fast_frame,
                sequential_frame,
                atol=1.5 / 255.0,
            )
            self.assertEqual(fast_frame[0, 0, 3], 0.0)
        await self._assert_reset_state(expected_state)

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
                const originalTotalTime = window.tl_to_use.totalTime;
                window.tl_to_use.totalTime = function(time, suppressEvents) {
                    if (arguments.length === 0) {
                        return originalTotalTime.call(this);
                    }
                    if (suppressEvents === false && time > 0) {
                        throw new Error("injected failure");
                    }
                    return originalTotalTime.call(this, time, suppressEvents);
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
            """() => {
                tl_to_use.reversed(true);
                tl_to_use.paused(true);
                const state = {
                    targetTime: tl_to_use.totalTime(),
                    targetPaused: tl_to_use.paused(),
                    targetReversed: tl_to_use.reversed(),
                };
                seekAndAppendToDomUsingTimes([0, 0.75], 128, false);
                return state;
            }"""
        )
        update_count_before_reset = await self.page.evaluate(
            "targetUpdateCount"
        )
        self.assertEqual(
            await self.page.locator("body > svg #shape").get_attribute(
                "data-progress"
            ),
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

    async def test_main_data_transform_sequences_match_direct_geometry(
        self,
    ) -> None:
        await self.page.add_script_tag(path=str(GSAP_JS))
        result = await self.page.evaluate(
            """() => {
                const shape = document.querySelector("#shape");
                window.tl_to_use = gsap.timeline({paused: true});
                tl_to_use.to(shape, {
                    x: 10,
                    y: 20,
                    rotation: 90,
                    scaleX: 2,
                    scaleY: 0.5,
                    skewX: 10,
                    transformOrigin: "10px 10px",
                    duration: 1,
                    ease: "none",
                });
                const direct = [];
                for (const time of [0, 0.5, 1]) {
                    tl_to_use.totalTime(time, false).pause();
                    const matrix = shape.getCTM();
                    const box = shape.getBBox();
                    const point = (x, y) => [
                        x * matrix.a + y * matrix.c + matrix.e,
                        x * matrix.b + y * matrix.d + matrix.f,
                    ];
                    direct.push({
                        CTM: [
                            [matrix.a, matrix.c, matrix.e],
                            [matrix.b, matrix.d, matrix.f],
                            [0, 0, 1],
                        ],
                        transformedPts: [
                            point(box.x, box.y),
                            point(box.x + box.width, box.y),
                            point(
                                box.x + box.width,
                                box.y + box.height,
                            ),
                            point(box.x, box.y + box.height),
                        ],
                    });
                }
                tl_to_use.totalTime(0, false).pause();
                return {
                    data: getAllTransformationValues([shape], 2),
                    direct,
                };
            }"""
        )

        data = result["data"]
        shape = data["shape"]
        self.assertEqual(
            data["info"],
            {
                "duration": 1,
                "fps": 2,
                "steps": 2,
                "scene-size": {"width": 20, "height": 20},
            },
        )
        expected_accumulated = {
            "translateX_acc": [0, 5, 10],
            "translateY_acc": [0, -10, -20],
            "rotate_acc": [0, 45, 90],
            "scaleX_acc": [1, 1.5, 2],
            "scaleY_acc": [1, 0.75, 0.5],
            "skewX_acc": [0, 5, 10],
            "skewY_acc": [0, 0, 0],
        }
        for name, expected in expected_accumulated.items():
            for actual, expected_value in zip(shape[name], expected):
                self.assertAlmostEqual(actual, expected_value, places=5)
        self.assertEqual(shape["transformTypes"], ["", "TRSK", "TRSK"])
        self.assertEqual(len(shape["tweens"]), 1)
        self.assertEqual(shape["tweens"][0]["start"], 0)
        self.assertEqual(shape["tweens"][0]["end"], 2)
        for frame_index, direct in enumerate(result["direct"]):
            for actual_row, expected_row in zip(
                shape["CTM"][frame_index],
                direct["CTM"],
            ):
                for actual, expected in zip(actual_row, expected_row):
                    self.assertAlmostEqual(actual, expected, places=5)
            for actual_point, expected_point in zip(
                shape["transformedPts"][frame_index],
                direct["transformedPts"],
            ):
                for actual, expected in zip(actual_point, expected_point):
                    self.assertAlmostEqual(actual, expected, places=5)

    async def test_animated_property_discovery_resolves_aliases_and_attrs(
        self,
    ) -> None:
        await self.page.add_script_tag(path=str(GSAP_JS))
        properties = await self.page.evaluate(
            """registry => {
                const source = document.querySelector("#source");
                const shape = document.querySelector("#shape");
                const namespace = "http://www.w3.org/2000/svg";
                const nestedGroup = document.createElementNS(namespace, "g");
                nestedGroup.id = "nested-group";
                const nestedShape = document.createElementNS(
                    namespace,
                    "circle",
                );
                nestedShape.id = "nested-shape";
                nestedShape.setAttribute("cx", "5");
                nestedShape.setAttribute("cy", "5");
                nestedShape.setAttribute("r", "2");
                nestedGroup.appendChild(nestedShape);
                source.appendChild(nestedGroup);
                window.tl_to_use = gsap.timeline({paused: true});
                tl_to_use.set(shape, {fill: "red"}, 0);
                tl_to_use.set(nestedShape, {fill: "blue"}, 0);
                tl_to_use.to(shape, {
                    x: 10,
                    y: 20,
                    rotation: 30,
                    scale: 2,
                    autoAlpha: 0.5,
                    attr: {r: 4, cx: 12},
                    duration: 1,
                    ease: "none",
                });
                tl_to_use.to(nestedShape, {
                    x: 10,
                    y: 20,
                    rotation: 30,
                    scale: 2,
                    autoAlpha: 0.5,
                    attr: {r: 4, cx: 12},
                    duration: 1,
                    ease: "none",
                }, 0);
                return {
                    animatedIds: getAllAnimatedElements(source).map(
                        element => element.id
                    ),
                    properties: extractAnimatedProperties(source, registry),
                };
            }""",
            PROPERTY_REGISTRY,
        )

        expected = {
            "cx",
            "fill",
            "opacity",
            "r",
            "rotate",
            "scaleX",
            "scaleY",
            "transformedPts",
            "translateX",
            "translateY",
            "visibility",
        }
        self.assertEqual(
            set(properties["animatedIds"]),
            {"shape", "nested-shape"},
        )
        properties = properties["properties"]
        self.assertEqual(set(properties), {"shape", "nested-shape"})
        self.assertEqual(set(properties["shape"]), expected)
        self.assertEqual(set(properties["nested-shape"]), expected)

    async def test_anonymous_animated_elements_get_unique_json_keys(
        self,
    ) -> None:
        await self.page.add_script_tag(path=str(GSAP_JS))
        result = await self.page.evaluate(
            """registry => {
                const namespace = "http://www.w3.org/2000/svg";
                const source = document.querySelector("#source");
                const group = document.createElementNS(namespace, "g");
                const first = document.createElementNS(namespace, "circle");
                const second = document.createElementNS(namespace, "circle");
                [first, second].forEach((element, index) => {
                    element.setAttribute("cx", String(4 + index * 8));
                    element.setAttribute("cy", "4");
                    element.setAttribute("r", "2");
                    group.appendChild(element);
                });
                source.appendChild(group);
                window.tl_to_use = gsap.timeline({paused: true});
                tl_to_use.to(
                    [first, second],
                    {x: 10, duration: 1, ease: "none"}
                );
                const elements = getAllAnimatedElements(source);
                const config = {
                    spatial: ["transformedPts", "translateX"],
                    visual: [],
                    svgAttributes: [],
                };
                const data = createRenderedData(
                    elements,
                    registry,
                    config,
                    2,
                );
                const properties = extractAnimatedProperties(
                    source,
                    registry,
                );
                return {
                    data,
                    keys: elements.map(
                        element => getMoverElementDataId(element)
                    ),
                    properties,
                };
            }""",
            PROPERTY_REGISTRY,
        )

        keys = result["keys"]
        self.assertEqual(len(keys), 2)
        self.assertEqual(len(set(keys)), 2)
        self.assertTrue(all(key.startswith("__mover_") for key in keys))
        self.assertEqual(set(result["properties"]), set(keys))
        self.assertEqual(
            set(result["data"]["info"]["generated-element-keys"]),
            set(keys),
        )
        for key in keys:
            self.assertEqual(len(result["data"][key]["translateX"]), 3)
            self.assertEqual(
                result["data"][key]["translateX"],
                [0, 5, 10],
            )
            self.assertEqual(
                set(result["properties"][key]),
                {"transformedPts", "translateX"},
            )

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
