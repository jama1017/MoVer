import io
import importlib
import inspect
import json
import math
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
    _load_opencv,
    capture_frames_server_driven,
    capture_png_frames_at_times,
    convert_animation,
    create_video_from_frames,
    parse_args,
    run_conversion,
)
from mover.converter.raster_capture import (
    _plan_batch_chunk_size,
    _plan_batch_layout,
)


OPTIONAL_JSON_FIXTURE = """<!doctype html>
<html>
<head>
    <script src="./gsap.min.js"></script>
</head>
<body>
    <svg width="40" height="40" viewBox="0 0 40 40">
        <rect id="square" x="5" y="5" width="10" height="10" fill="black"/>
    </svg>
    <script>
        const square = document.getElementById("square");
        const tl = gsap.timeline({ paused: true });
        tl.to(square, { x: 10, duration: 0.1, ease: "none" });
    </script>
    <script src="./vis.js"></script>
    <script src="./convert.js"></script>
</body>
</html>
"""


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
        if "getAnimationInfo(fps)" in expression:
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

    async def test_invalid_capture_duration_is_rejected_before_page_work(self) -> None:
        page = FakePage()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "missing" / "frames"
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                await capture_frames_server_driven(
                    page,
                    str(output_path),
                    output_format="png",
                    in_memory=True,
                    capture_duration=float("inf"),
                )

            self.assertEqual(page.evaluate_calls, [])
            self.assertFalse(output_path.parent.exists())

    async def test_capture_duration_is_forwarded_to_browser_preparation(self) -> None:
        page = FakePage()

        with tempfile.TemporaryDirectory() as temp_dir:
            await capture_frames_server_driven(
                page,
                str(Path(temp_dir) / "frames"),
                fps=30,
                output_format="png",
                in_memory=True,
                capture_duration=2.5,
            )

        self.assertEqual(page.evaluate_calls[0][1], [30, 2.5])

    async def test_batched_capture_requires_paired_dimensions_before_page_work(
        self,
    ) -> None:
        page = FakePage()
        with self.assertRaisesRegex(
            ValueError,
            "requires explicit width and height",
        ):
            await capture_frames_server_driven(
                page,
                "unused",
                output_format="png",
                capture_strategy="batched",
            )
        with self.assertRaisesRegex(
            ValueError,
            "provided together",
        ):
            await capture_frames_server_driven(
                page,
                "unused",
                output_format="png",
                width=320,
            )
        self.assertEqual(page.evaluate_calls, [])

    async def test_invalid_explicit_capture_request_precedes_page_work(
        self,
    ) -> None:
        page = FakePage()
        for kwargs, message in (
            ({"strategy": "unknown"}, "Unsupported capture strategy"),
            ({"width": 0, "height": 10}, "positive integers"),
            ({"seek_times": [float("nan")]}, "finite numbers"),
        ):
            seek_times = kwargs.pop("seek_times", [0.0])
            with self.assertRaisesRegex(ValueError, message):
                await capture_png_frames_at_times(
                    page,
                    seek_times,
                    **kwargs,
                )
        self.assertEqual(page.evaluate_calls, [])

    def test_batch_chunk_limit_is_dimension_and_pixel_bounded(self) -> None:
        self.assertEqual(_plan_batch_chunk_size(128, 128), 100)
        self.assertEqual(_plan_batch_chunk_size(1920, 1080), 3)
        self.assertEqual(_plan_batch_chunk_size(20_000, 1), 0)
        self.assertEqual(_plan_batch_chunk_size(8_000, 8_000), 0)
        self.assertEqual(
            _plan_batch_layout(320, 180, 1280, 720),
            (16, 4),
        )
        capacity, columns = _plan_batch_layout(
            1000,
            1000,
            7000,
            2000,
        )
        self.assertEqual((capacity, columns), (8, 4))
        rows = math.ceil(capacity / columns)
        self.assertLessEqual(
            columns * 1000 * rows * 1000,
            8_000_000,
        )

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
        for function in (
            capture_frames_server_driven,
            run_conversion,
            convert_animation,
        ):
            self.assertIsNone(
                inspect.signature(function)
                .parameters["capture_duration"]
                .default
            )
            self.assertEqual(
                inspect.signature(function)
                .parameters["capture_strategy"]
                .default,
                "sequential",
            )
            self.assertIsNone(
                inspect.signature(function).parameters["width"].default
            )
            self.assertIsNone(
                inspect.signature(function).parameters["height"].default
            )
            self.assertFalse(
                inspect.signature(function)
                .parameters["omit_background"]
                .default
            )
        with patch("sys.argv", ["mover-converter", "example.html", "0"]):
            args = parse_args()
            self.assertEqual(args.video_fps, DEFAULT_FPS)
            self.assertIsNone(args.capture_duration)
            self.assertEqual(args.capture_strategy, "sequential")
            self.assertIsNone(args.width)
            self.assertIsNone(args.height)
            self.assertFalse(args.omit_background)

    def test_capture_duration_cli_option(self) -> None:
        with patch(
            "sys.argv",
            [
                "mover-converter",
                "example.html",
                "0",
                "--capture-duration",
                "2.5",
            ],
        ):
            self.assertEqual(parse_args().capture_duration, 2.5)

    def test_convert_rejects_invalid_capture_duration_before_server_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            convert_animation(
                "unused.html",
                port=0,
                capture_duration=0,
            )

    def test_convert_rejects_invalid_batched_request_before_server_start(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires explicit width and height",
        ):
            convert_animation(
                "unused.html",
                port=0,
                create_video=True,
                output_format="png",
                capture_strategy="batched",
            )

    def test_batched_cli_options(self) -> None:
        with patch(
            "sys.argv",
            [
                "mover-converter",
                "example.html",
                "0",
                "--capture-strategy",
                "batched",
                "--width",
                "320",
                "--height",
                "180",
                "--omit-background",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.capture_strategy, "batched")
        self.assertEqual(args.width, 320)
        self.assertEqual(args.height, 180)
        self.assertTrue(args.omit_background)

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

    @patch("mover.converter.mover_converter.subprocess.run")
    @unittest.skipUnless(
        importlib.util.find_spec("cv2") is not None,
        "OpenCV fallback requires mover[media] or mover[full]",
    )
    def test_mp4_keeps_valid_opencv_output_when_ffmpeg_fails(
        self,
        mock_run,
    ) -> None:
        mock_run.side_effect = [
            subprocess.CompletedProcess(["ffmpeg", "-version"], 0),
            subprocess.CalledProcessError(1, ["ffmpeg"]),
        ]
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "animation.mp4"
            create_video_from_frames(
                [frame],
                str(output_path),
                fps=5,
                output_format="mp4",
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertFalse((Path(temp_dir) / "animation.temp.mp4").exists())
            self.assertFalse((Path(temp_dir) / "animation.ffmpeg.mp4").exists())

            cv2 = importlib.import_module("cv2")
            capture = cv2.VideoCapture(str(output_path))
            try:
                self.assertTrue(capture.isOpened())
                decoded, decoded_frame = capture.read()
                self.assertTrue(decoded)
                self.assertIsNotNone(decoded_frame)
            finally:
                capture.release()

    @patch(
        "mover.converter.mover_converter.subprocess.run",
        side_effect=FileNotFoundError,
    )
    @patch("mover.converter.mover_converter._load_opencv")
    def test_mp4_reports_opencv_writer_failure(
        self,
        mock_load_opencv,
        _mock_run,
    ) -> None:
        mock_load_opencv.return_value.VideoWriter.return_value.isOpened.return_value = (
            False
        )
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "animation.mp4"
            with self.assertRaisesRegex(
                RuntimeError,
                "could not initialize",
            ):
                create_video_from_frames(
                    [frame],
                    str(output_path),
                    fps=5,
                    output_format="mp4",
                )

            self.assertFalse(output_path.exists())
            self.assertFalse((Path(temp_dir) / "animation.temp.mp4").exists())

    @patch("mover.converter.mover_converter._load_opencv")
    @patch("mover.converter.mover_converter.subprocess.run")
    def test_ffmpeg_mp4_does_not_load_opencv(
        self,
        mock_run,
        mock_load_opencv,
    ) -> None:
        def run_ffmpeg(command, **kwargs):
            if "-version" not in command and "-f" in command and "rawvideo" in command:
                Path(command[-1]).write_bytes(b"fake-mp4")
            return subprocess.CompletedProcess(command, 0)

        mock_run.side_effect = run_ffmpeg
        mock_load_opencv.side_effect = AssertionError("OpenCV should not load")
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "animation.mp4"
            create_video_from_frames(
                [frame],
                str(output_path),
                fps=5,
                output_format="mp4",
            )

            self.assertEqual(output_path.read_bytes(), b"fake-mp4")
            mock_load_opencv.assert_not_called()

    def test_missing_opencv_recommends_media_extra(self) -> None:
        with patch.dict("sys.modules", {"cv2": None}):
            with self.assertRaisesRegex(RuntimeError, r"mover\[media\]"):
                _load_opencv()

    @patch(
        "mover.converter.mover_converter.subprocess.run",
        side_effect=FileNotFoundError,
    )
    @patch(
        "mover.converter.mover_converter._load_opencv",
        side_effect=RuntimeError(
            'Install FFmpeg or run: pip install "mover[media]"'
        ),
    )
    def test_missing_mp4_fallback_recommends_media_extra(
        self,
        _mock_load_opencv,
        _mock_run,
    ) -> None:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "animation.mp4"
            with self.assertRaisesRegex(RuntimeError, r"mover\[media\]"):
                create_video_from_frames(
                    [frame],
                    str(output_path),
                    fps=5,
                    output_format="mp4",
                )
            self.assertFalse(output_path.exists())


class OptionalJsonOutputIntegrationTest(unittest.TestCase):
    def test_opt_in_batched_png_matches_exact_sequential_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            html_path = root / "raster.html"
            html_path.write_text(OPTIONAL_JSON_FIXTURE, encoding="utf-8")
            outputs: dict[str, list[np.ndarray]] = {}
            for strategy in ("sequential", "batched"):
                output_dir = root / strategy
                convert_animation(
                    str(html_path),
                    port=0,
                    create_video=True,
                    output_format="png",
                    video_fps=10,
                    output_dir=str(output_dir),
                    hide_grid=True,
                    capture_strategy=strategy,
                    width=80,
                    height=40,
                )
                frame_paths = sorted(
                    (output_dir / "raster_animation_10_png").glob(
                        "frame_*.png"
                    )
                )
                self.assertEqual(len(frame_paths), 2)
                outputs[strategy] = [
                    np.asarray(
                        Image.open(path).convert("RGBA"),
                        dtype=np.float32,
                    )
                    / 255.0
                    for path in frame_paths
                ]
                self.assertTrue(
                    all(frame.shape == (40, 80, 4) for frame in outputs[strategy])
                )

            for sequential, batched in zip(
                outputs["sequential"],
                outputs["batched"],
            ):
                np.testing.assert_allclose(
                    batched,
                    sequential,
                    atol=1.5 / 255.0,
                )

    def test_all_documented_json_outputs_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            html_path = root / "optional_json.html"
            output_dir = root / "output"
            html_path.write_text(OPTIONAL_JSON_FIXTURE, encoding="utf-8")

            convert_animation(
                str(html_path),
                port=0,
                save_keyframes=True,
                save_for_comparison=True,
                save_animated_properties=True,
                output_dir=str(output_dir),
                video_fps=10,
            )

            expected_files = {
                "optional_json_data.json",
                "optional_json_data_keyframes.json",
                "optional_json_data_rendered.json",
                "optional_json_properties.json",
            }
            self.assertEqual(
                {path.name for path in output_dir.glob("*.json")},
                expected_files,
            )

            main_data = json.loads(
                (output_dir / "optional_json_data.json").read_text(
                    encoding="utf-8"
                )
            )
            rendered_data = json.loads(
                (output_dir / "optional_json_data_rendered.json").read_text(
                    encoding="utf-8"
                )
            )
            properties = json.loads(
                (output_dir / "optional_json_properties.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(main_data["info"]["fps"], 10)
            self.assertEqual(rendered_data["info"]["fps"], 10)
            self.assertIn("square", properties)
            self.assertTrue(properties["square"])


if __name__ == "__main__":
    unittest.main()
