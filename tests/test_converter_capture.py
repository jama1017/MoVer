import io
import inspect
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from mover.converter.mover_converter import (
    DEFAULT_FPS,
    _get_animation_output_path,
    capture_frames_server_driven,
    convert_animation,
    create_video_from_frames,
    parse_args,
    run_conversion,
)


def make_png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (2, 1), (64, 128, 255, 32)).save(output, format="PNG")
    return output.getvalue()


class FakeSvgLocator:
    def __init__(self, png_bytes: bytes) -> None:
        self.first = self
        self.png_bytes = png_bytes
        self.screenshot_calls: list[dict] = []
        self.evaluate_calls: list[tuple[str, object | None]] = []
        self.style = "display: block"

    async def wait_for(self, **kwargs) -> None:
        return None

    async def screenshot(self, **kwargs) -> bytes:
        self.screenshot_calls.append(kwargs)
        return self.png_bytes

    async def get_attribute(self, name: str):
        if name != "style":
            raise AssertionError(f"Unexpected attribute: {name}")
        return self.style

    async def evaluate(self, expression: str, argument=None):
        self.evaluate_calls.append((expression, argument))
        if "background-image" in expression:
            self.style = f"{self.style}; background-image: none !important"
        elif "originalStyle" in expression:
            self.style = argument


class FakePage:
    def __init__(self) -> None:
        self.svg_locator = FakeSvgLocator(make_png_bytes())
        self.evaluate_calls: list[tuple[str, object | None]] = []

    def locator(self, selector: str) -> FakeSvgLocator:
        if selector != "svg":
            raise AssertionError(f"Unexpected locator: {selector}")
        return self.svg_locator

    async def evaluate(self, expression: str, argument=None):
        self.evaluate_calls.append((expression, argument))
        if expression == "fps => getAnimationInfo(fps)":
            return {"animDuration": 1.25, "steps": 1}
        if "XMLSerializer" in expression:
            return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        return None


class CaptureFramesServerDrivenTest(unittest.IsolatedAsyncioTestCase):
    async def test_in_memory_png_returns_frames_and_duration_without_disk_io(self) -> None:
        page = FakePage()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "missing" / "frames"
            frames, duration = await capture_frames_server_driven(
                page,
                str(output_path),
                fps=30,
                output_format="png",
                in_memory=True,
            )

            self.assertEqual(duration, 1.25)
            self.assertEqual(len(frames), 2)
            self.assertFalse(output_path.parent.exists())
            self.assertEqual(page.svg_locator.screenshot_calls, [{"type": "png"}] * 2)
            for frame in frames:
                self.assertIsInstance(frame, np.ndarray)
                self.assertEqual(frame.shape, (1, 2, 4))
                self.assertEqual(frame.dtype, np.float32)

    async def test_in_memory_svg_matches_disk_text_without_disk_io(self) -> None:
        page = FakePage()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "missing" / "frames"
            frames, duration = await capture_frames_server_driven(
                page,
                str(output_path),
                fps=30,
                output_format="svg",
                in_memory=True,
            )

            self.assertEqual(duration, 1.25)
            self.assertEqual(len(frames), 2)
            self.assertFalse(output_path.parent.exists())
            self.assertEqual(page.svg_locator.screenshot_calls, [])
            for frame in frames:
                self.assertIsInstance(frame, io.StringIO)
                self.assertEqual(
                    frame.getvalue(),
                    '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
                )

    async def test_in_memory_video_is_rejected_before_browser_or_disk_work(self) -> None:
        page = FakePage()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "missing" / "animation.mp4"
            with self.assertRaisesRegex(ValueError, "only supported for PNG and SVG"):
                await capture_frames_server_driven(
                    page,
                    str(output_path),
                    output_format="mp4",
                    in_memory=True,
                )

            self.assertEqual(page.evaluate_calls, [])
            self.assertFalse(output_path.parent.exists())

    async def test_disk_png_uses_element_screenshot_bytes(self) -> None:
        page = FakePage()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "frames"
            result = await capture_frames_server_driven(
                page,
                str(output_path),
                fps=30,
                output_format="png",
            )

            self.assertIsNone(result)
            self.assertEqual(page.svg_locator.screenshot_calls, [{"type": "png"}] * 2)
            self.assertEqual(
                sorted(path.read_bytes() for path in output_path.glob("frame_*.png")),
                [page.svg_locator.png_bytes] * 2,
            )

    @patch(
        "mover.converter.mover_converter.subprocess.run",
        side_effect=FileNotFoundError,
    )
    async def test_disk_gif_requires_ffmpeg(self, _mock_run) -> None:
        page = FakePage()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "animation.gif"
            with self.assertRaisesRegex(RuntimeError, "requires a working FFmpeg"):
                await capture_frames_server_driven(
                    page,
                    str(output_path),
                    fps=30,
                    output_format="gif",
                )
            self.assertFalse(output_path.exists())

    async def test_hide_grid_restores_inline_style_after_screenshot(self) -> None:
        page = FakePage()
        original_style = page.svg_locator.style

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "missing" / "frames"
            frames, duration = await capture_frames_server_driven(
                page,
                str(output_path),
                output_format="png",
                in_memory=True,
                hide_grid=True,
            )

            self.assertEqual(duration, 1.25)
            self.assertEqual(len(frames), 2)
            self.assertFalse(output_path.parent.exists())
            self.assertEqual(
                page.svg_locator.screenshot_calls,
                [{"type": "png"}] * 2,
            )
            self.assertEqual(page.svg_locator.style, original_style)
            self.assertEqual(len(page.svg_locator.evaluate_calls), 4)


class OutputNamingTest(unittest.TestCase):
    def test_public_default_fps_remains_60(self) -> None:
        self.assertEqual(DEFAULT_FPS, 60)
        for function, parameter_name in (
            (create_video_from_frames, "fps"),
            (capture_frames_server_driven, "fps"),
            (run_conversion, "video_fps"),
            (convert_animation, "video_fps"),
        ):
            self.assertEqual(
                inspect.signature(function).parameters[parameter_name].default,
                DEFAULT_FPS,
            )
        with patch("sys.argv", ["mover-converter", "example.html", "0"]):
            self.assertEqual(parse_args().video_fps, DEFAULT_FPS)

    def test_video_names_remain_legacy_compatible(self) -> None:
        self.assertEqual(
            _get_animation_output_path("/tmp/output", "example", "mp4", 60),
            Path("/tmp/output/example_animation.mp4"),
        )
        self.assertEqual(
            _get_animation_output_path("/tmp/output", "example", "GIF", 24),
            Path("/tmp/output/example_animation.gif"),
        )

    def test_frame_directories_include_fps(self) -> None:
        self.assertEqual(
            _get_animation_output_path("/tmp/output", "example", "png", 30),
            Path("/tmp/output/example_animation_30_png"),
        )
        self.assertEqual(
            _get_animation_output_path("/tmp/output", "example", "SVG", 12),
            Path("/tmp/output/example_animation_12_svg"),
        )

    def test_unsupported_output_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported output format"):
            _get_animation_output_path("/tmp/output", "example", "webm", 30)

    def test_hide_grid_cli_flag_defaults_off_and_can_be_enabled(self) -> None:
        with patch("sys.argv", ["mover-converter", "example.html", "0"]):
            self.assertFalse(parse_args().hide_grid)
        with patch("sys.argv", ["mover-converter", "example.html", "0", "--hide-grid"]):
            self.assertTrue(parse_args().hide_grid)

    @patch(
        "mover.converter.mover_converter.subprocess.run",
        side_effect=FileNotFoundError,
    )
    def test_gif_requires_ffmpeg_when_unavailable(self, _mock_run) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "animation.gif"
            with self.assertRaisesRegex(RuntimeError, "requires a working FFmpeg"):
                create_video_from_frames(
                    [frame],
                    str(output_path),
                    fps=5,
                    output_format="gif",
                )
            self.assertFalse(output_path.exists())

    @patch("mover.converter.mover_converter.subprocess.run")
    def test_gif_reports_ffmpeg_encoding_failure(self, mock_run) -> None:
        mock_run.side_effect = [
            subprocess.CompletedProcess(["ffmpeg", "-version"], 0),
            subprocess.CalledProcessError(1, ["ffmpeg"]),
        ]
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "animation.gif"
            with self.assertRaisesRegex(RuntimeError, "GIF palette support"):
                create_video_from_frames(
                    [frame],
                    str(output_path),
                    fps=5,
                    output_format="gif",
                )
            self.assertFalse(output_path.exists())
            self.assertFalse((Path(temp_dir) / "animation.temp.mp4").exists())

if __name__ == "__main__":
    unittest.main()
