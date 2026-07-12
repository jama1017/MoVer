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
                svg {
                    background-color: white;
                    background-image: linear-gradient(rgb(255, 0, 0), rgb(255, 0, 0));
                    background-repeat: no-repeat;
                }
            </style>
            <body style="padding: 3px 4px !important; background: white">
                <p id="prompt" style="display: inline-block !important; color: blue">Prompt</p>
                <svg id="source" width="20" height="20" style="display: inline">
                    <style>#shape { stroke: black; }</style>
                    <defs>
                        <linearGradient id="paint">
                            <stop stop-color="green"/>
                        </linearGradient>
                    </defs>
                    <circle id="shape" cx="10" cy="10" r="3" fill="url(#paint)"/>
                </svg>
                <br id="break" style="display: block">
                <div id="existing" style="display: flex !important">Keep me</div>
                <script>
                    window.tl_to_use = {
                        seek(time) { this.time = time; },
                        pause() {},
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
                bodyStyle: document.body.getAttribute("style"),
                elements: Array.from(document.body.children)
                    .filter(element => element.tagName !== "SCRIPT")
                    .map(element => ({
                        id: element.id,
                        style: element.getAttribute("style"),
                    })),
            })"""
        )

    async def _assert_reset_state(self) -> None:
        self.assertEqual(await self._snapshot_original_state(), self.initial_state)
        self.assertEqual(
            await self.page.locator("body > [data-mover-batch-frame]").count(),
            0,
        )

    async def test_frame_sizes_grid_control_ids_and_reset(self) -> None:
        await self.page.set_viewport_size({"width": 128, "height": 256})
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
                    rootId: svg.id,
                    paintId: svg.querySelector("linearGradient").id,
                    fill: svg.querySelector("circle").getAttribute("fill"),
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
            [state["rootId"] for state in default_state],
            ["mover_frame_0_source", "mover_frame_1_source"],
        )
        self.assertEqual(
            [state["paintId"] for state in default_state],
            ["mover_frame_0_paint", "mover_frame_1_paint"],
        )
        self.assertEqual(
            [state["fill"] for state in default_state],
            ["url(#mover_frame_0_paint)", "url(#mover_frame_1_paint)"],
        )
        self.assertEqual(
            [state["styleText"].strip() for state in default_state],
            [
                "#mover_frame_0_shape { stroke: black; }",
                "#mover_frame_1_shape { stroke: black; }",
            ],
        )

        screenshot = Image.open(
            io.BytesIO(await self.page.screenshot(type="png", full_page=True))
        )
        self.assertEqual(screenshot.size, (128, 256))
        self.assertEqual(screenshot.convert("RGB").getpixel((1, 1)), (255, 0, 0))

        self.assertTrue(await self.page.evaluate("resetSeekAndAppend()"))
        await self._assert_reset_state()

        await self.page.set_viewport_size({"width": 512, "height": 1024})
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
            ["mover_frame_0_source", "mover_frame_1_source"],
        )

        screenshot = Image.open(
            io.BytesIO(await self.page.screenshot(type="png", full_page=True))
        )
        self.assertEqual(screenshot.size, (512, 1024))
        self.assertEqual(screenshot.convert("RGB").getpixel((1, 1)), (255, 255, 255))

        self.assertTrue(await self.page.evaluate("resetSeekAndAppend()"))
        await self._assert_reset_state()

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
        with self.assertRaisesRegex(Exception, "injected failure"):
            await self.page.evaluate("seekAndAppendToDomUsingTimes([0, 1])")
        await self._assert_reset_state()
        self.assertFalse(await self.page.evaluate("resetSeekAndAppend()"))

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


if __name__ == "__main__":
    unittest.main()
