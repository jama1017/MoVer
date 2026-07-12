import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from mover.converter.mover_converter import capture_frames_server_driven


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
                body > svg .shape-resource {
                    fill: url("#paint");
                }
                body > svg #symbol .part {
                    fill: rgb(0, 0, 255);
                    opacity: 0.4;
                }
                #prompt + svg #shape {
                    stroke-width: 3px;
                }
                svg:first-of-type #shape {
                    stroke-linecap: round;
                }
            </style>
            <body style="padding: 3px 4px !important; background: white">
                <p id="prompt" onclick="window.parent.hacked = true" style="display: inline-block !important; color: blue">Prompt</p>
                <svg id="source" width="20" height="20" style="display: inline">
                    <style>#shape { stroke: black; }</style>
                    <defs>
                        <linearGradient id="paint">
                            <stop stop-color="green"/>
                        </linearGradient>
                    </defs>
                    <circle class="shape-resource" id="shape" cx="10" cy="10" r="3"/>
                    <g id="symbol">
                        <rect class="part" x="15" y="15" width="2" height="2"/>
                    </g>
                    <use id="symbol-copy" href="#symbol" x="8"/>
                </svg>
                <br id="break" style="display: block">
                <div id="existing" style="display: flex !important">Keep me</div>
                <script>
                    window.gsap = {
                        globalTimeline: {
                            pauseCalls: [],
                            timeCalls: [],
                            currentTime: 4,
                            isPaused: false,
                            pause() {
                                this.pauseCalls.push(
                                    arguments.length ? arguments[0] : null
                                );
                                this.isPaused = true;
                                return this;
                            },
                            time(time, suppressEvents) {
                                if (arguments.length === 0) {
                                    return this.currentTime;
                                }
                                this.timeCalls.push([time, suppressEvents]);
                                this.currentTime = time;
                                window.ambientRenderedTime = time - 8;
                                return this;
                            },
                            paused(value) {
                                if (arguments.length === 0) {
                                    return this.isPaused;
                                }
                                this.isPaused = value;
                                return this;
                            },
                            play() {
                                this.isPaused = false;
                                return this;
                            },
                        },
                    };
                    window.tl_to_use = {
                        seekCalls: [],
                        globalTimeCalls: [],
                        currentTotalTime: 1,
                        isPaused: true,
                        seek(time) {
                            this.seekCalls.push(time);
                            this.renderedTime = time;
                            this.currentTotalTime = time;
                        },
                        globalTime(time) {
                            this.globalTimeCalls.push(time);
                            return 10 + time;
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
            await self.page.locator("body > [data-mover-batch-frame]").count(),
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
                document.querySelectorAll("body > [data-mover-batch-frame]")
            ).map(frame => {
                const rect = frame.getBoundingClientRect();
                const svg = frame.contentDocument.querySelector("body > svg");
                return {
                    rect: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                    },
                    sandbox: frame.getAttribute("sandbox"),
                    scriptCount: frame.contentDocument.querySelectorAll("script").length,
                    eventHandlerCount: Array.from(
                        frame.contentDocument.querySelectorAll("*")
                    ).reduce(
                        (count, element) => count + element.getAttributeNames()
                            .filter(attr => attr.toLowerCase().startsWith("on")).length,
                        0,
                    ),
                    freezeStyleCount: frame.contentDocument.querySelectorAll(
                        "style[data-mover-snapshot-freeze]"
                    ).length,
                    svgCount: frame.contentDocument.querySelectorAll("body > svg").length,
                    backgroundImage: getComputedStyle(svg).backgroundImage,
                    rootId: svg.id,
                    paintId: svg.querySelector("linearGradient").id,
                    shapeId: svg.querySelector("circle").id,
                    shapeOpacity: getComputedStyle(svg.querySelector("circle")).opacity,
                    shapeStrokeWidth: getComputedStyle(svg.querySelector("circle")).strokeWidth,
                    shapeStrokeLinecap: getComputedStyle(svg.querySelector("circle")).strokeLinecap,
                    fill: getComputedStyle(svg.querySelector("circle")).fill,
                    symbolId: svg.querySelector("g").id,
                    symbolUseHref: svg.querySelector("use").getAttribute("href"),
                    partFill: getComputedStyle(svg.querySelector(".part")).fill,
                    partOpacity: getComputedStyle(svg.querySelector(".part")).opacity,
                    styleText: svg.querySelector(
                        "style:not([data-mover-rewritten-resource-styles])"
                    ).textContent,
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
        self.assertEqual(
            [state["sandbox"] for state in default_state],
            ["allow-same-origin", "allow-same-origin"],
        )
        self.assertTrue(all(state["scriptCount"] >= 1 for state in default_state))
        self.assertEqual([state["eventHandlerCount"] for state in default_state], [1, 1])
        self.assertEqual([state["freezeStyleCount"] for state in default_state], [0, 0])
        self.assertEqual(
            await self.page.evaluate("typeof window.hacked"),
            "undefined",
        )
        self.assertTrue(
            all("linear-gradient" in state["backgroundImage"] for state in default_state)
        )
        self.assertEqual(
            [state["rootId"] for state in default_state],
            ["source", "source"],
        )
        self.assertEqual(
            [state["paintId"] for state in default_state],
            ["paint", "paint"],
        )
        self.assertEqual(
            [state["shapeId"] for state in default_state],
            ["shape", "shape"],
        )
        self.assertEqual(
            [state["shapeOpacity"] for state in default_state],
            ["0.5", "0.5"],
        )
        self.assertEqual(
            [state["shapeStrokeWidth"] for state in default_state],
            ["3px", "3px"],
        )
        self.assertEqual(
            [state["shapeStrokeLinecap"] for state in default_state],
            ["round", "round"],
        )
        self.assertIn("#paint", default_state[0]["fill"])
        self.assertIn("#paint", default_state[1]["fill"])
        self.assertEqual(
            [state["symbolId"] for state in default_state],
            ["symbol", "symbol"],
        )
        self.assertEqual(
            [state["symbolUseHref"] for state in default_state],
            ["#symbol", "#symbol"],
        )
        self.assertEqual(
            [state["partFill"] for state in default_state],
            ["rgb(0, 0, 255)", "rgb(0, 0, 255)"],
        )
        self.assertEqual(
            [state["partOpacity"] for state in default_state],
            ["0.4", "0.4"],
        )
        self.assertEqual(
            [state["styleText"].strip() for state in default_state],
            [
                "#shape { stroke: black; }",
                "#shape { stroke: black; }",
            ],
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
                rootTime: gsap.globalTimeline.time(),
                rootPaused: gsap.globalTimeline.paused(),
                targetTime: tl_to_use.totalTime(),
                targetPaused: tl_to_use.paused(),
            })"""
        )
        self.assertEqual(
            restored_gsap_state,
            {
                "rootTime": 4,
                "rootPaused": False,
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
            ).map(frame => {
                const rect = frame.getBoundingClientRect();
                const svg = frame.contentDocument.querySelector("body > svg");
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
                window.gsap.globalTimeline.time = time => {
                    if (time > 10) throw new Error("injected failure");
                };
            }"""
        )
        expected_failure_state = await self._snapshot_original_state()
        with self.assertRaisesRegex(Exception, "injected failure"):
            await self.page.evaluate("seekAndAppendToDomUsingTimes([0, 1])")
        await self._assert_reset_state(expected_failure_state)
        self.assertFalse(await self.page.evaluate("resetSeekAndAppend()"))

    async def test_seek_synchronizes_target_and_sibling_global_animations(self) -> None:
        await self.page.evaluate("seekToTime(2.5)")
        synchronized_state = await self.page.evaluate(
            """() => ({
                globalTimeCalls: tl_to_use.globalTimeCalls,
                targetRenderedTime: tl_to_use.renderedTime,
                ambientRenderedTime,
                rootTimeCalls: gsap.globalTimeline.timeCalls,
                rootPauseCalls: gsap.globalTimeline.pauseCalls,
                directSeekCalls: tl_to_use.seekCalls,
            })"""
        )
        self.assertEqual(synchronized_state["globalTimeCalls"], [2.5])
        self.assertEqual(synchronized_state["targetRenderedTime"], 2.5)
        self.assertEqual(synchronized_state["ambientRenderedTime"], 4.5)
        self.assertEqual(synchronized_state["rootTimeCalls"], [[12.5, False]])
        self.assertEqual(synchronized_state["rootPauseCalls"], [None, None])
        self.assertEqual(synchronized_state["directSeekCalls"], [2.5])

        await self.page.evaluate(
            """() => {
                delete tl_to_use.globalTime;
                seekToTime(1.25);
            }"""
        )
        self.assertEqual(
            await self.page.evaluate("tl_to_use.seekCalls"),
            [2.5, 1.25],
        )

    async def test_real_gsap_synchronizes_and_restores_timelines(self) -> None:
        await self.page.add_script_tag(path=str(GSAP_JS))
        initial_state = await self.page.evaluate(
            """() => {
                window.targetObject = { value: 0 };
                window.siblingObject = { value: 0 };
                window.tl_to_use = gsap.timeline({ paused: true });
                tl_to_use.to(
                    targetObject,
                    { value: 100, duration: 1, ease: "none" }
                );
                window.targetUpdateCount = 0;
                tl_to_use.eventCallback(
                    "onUpdate",
                    () => targetUpdateCount++
                );
                tl_to_use.timeScale(2);
                gsap.to(
                    siblingObject,
                    { value: 200, duration: 1, ease: "none" }
                );
                return {
                    rootTime: gsap.globalTimeline.totalTime(),
                    rootPaused: gsap.globalTimeline.paused(),
                    targetTime: tl_to_use.totalTime(),
                    targetPaused: tl_to_use.paused(),
                };
            }"""
        )

        await self.page.evaluate("seekToTime(0.5)")
        synchronized = await self.page.evaluate(
            """() => ({
                targetValue: targetObject.value,
                siblingValue: siblingObject.value,
                rootTime: gsap.globalTimeline.totalTime(),
                targetTime: tl_to_use.totalTime(),
            })"""
        )
        self.assertAlmostEqual(synchronized["targetValue"], 50, places=4)
        self.assertGreater(synchronized["siblingValue"], 40)
        self.assertLess(synchronized["siblingValue"], 60)
        self.assertAlmostEqual(synchronized["targetTime"], 0.5, places=6)

        await self.page.evaluate(
            """() => {
                tl_to_use.reversed(true);
                tl_to_use.paused(false);
            }"""
        )
        before_batch = await self.page.evaluate(
            """() => ({
                rootTime: gsap.globalTimeline.totalTime(),
                rootPaused: gsap.globalTimeline.paused(),
                targetTime: tl_to_use.totalTime(),
                targetPaused: tl_to_use.paused(),
                targetReversed: tl_to_use.reversed(),
            })"""
        )
        await self.page.evaluate(
            "seekAndAppendToDomUsingTimes([0, 0.75], 128, false)"
        )
        update_count_before_reset = await self.page.evaluate(
            "targetUpdateCount"
        )
        self.assertTrue(await self.page.evaluate("resetSeekAndAppend()"))
        self.assertEqual(
            await self.page.evaluate("targetUpdateCount"),
            update_count_before_reset,
        )
        after_batch = await self.page.evaluate(
            """() => ({
                rootTime: gsap.globalTimeline.totalTime(),
                rootPaused: gsap.globalTimeline.paused(),
                targetTime: tl_to_use.totalTime(),
                targetPaused: tl_to_use.paused(),
                targetReversed: tl_to_use.reversed(),
            })"""
        )
        self.assertAlmostEqual(
            after_batch["rootTime"],
            before_batch["rootTime"],
            places=6,
        )
        self.assertEqual(after_batch["rootPaused"], before_batch["rootPaused"])
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
        self.assertTrue(initial_state["targetPaused"])

    async def test_animated_property_conversion_does_not_leak_global(self) -> None:
        result = await self.page.evaluate(
            """registry => convertAnimatedPropertiesToJson(registry)""",
            {
                "gsapAliases": {},
                "spatial": {},
                "visual": {},
                "svgAttributes": {},
            },
        )
        self.assertEqual(result, {})
        self.assertEqual(
            await self.page.evaluate("typeof window.animatedProps"),
            "undefined",
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
            pause_calls = await self.page.evaluate(
                "gsap.globalTimeline.pauseCalls"
            )
            self.assertGreaterEqual(len(pause_calls), 5)
            self.assertTrue(all(call is None for call in pause_calls))


if __name__ == "__main__":
    unittest.main()
