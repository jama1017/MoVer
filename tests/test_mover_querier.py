import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mover.converter.mover_querier import get_position_in_time, parse_args


SVGPT_QUERY_FIXTURE = """<!doctype html>
<html>
<head>
    <script src="./gsap.min.js"></script>
</head>
<body>
    <svg width="400" height="400" viewBox="0 0 400 400">
        <circle id="circle'quoted" cx="25" cy="25" r="5" fill="blue"/>
    </svg>
    <script>
        const circle = document.getElementById("circle'quoted");
        const tl = gsap.timeline({ paused: true });
        tl.to(circle, { x: 100, duration: 0.01, ease: "none" });
        let tl_to_use = tl;
    </script>
    <script src="./convert.js"></script>
</body>
</html>
"""


class MoverQuerierTest(unittest.TestCase):
    def test_positional_cli_remains_compatible(self) -> None:
        with patch(
            "sys.argv",
            [
                "mover-querier",
                "animation.html",
                "0",
                "125,25",
                "circle'quoted",
                "0.1",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.html_file, "animation.html")
        self.assertEqual(args.port, 0)
        self.assertEqual(args.target_centroids, "125,25")
        self.assertEqual(args.element_id, "circle'quoted")
        self.assertEqual(args.tolerance, 0.1)
        self.assertIsNone(args.capture_duration)

    def test_capture_duration_cli_option(self) -> None:
        with patch(
            "sys.argv",
            [
                "mover-querier",
                "animation.html",
                "0",
                "125,25",
                "circle",
                "0.1",
                "--capture-duration",
                "2.5",
            ],
        ):
            self.assertEqual(parse_args().capture_duration, 2.5)

    def test_invalid_capture_duration_fails_before_server_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            get_position_in_time(
                "unused.html",
                [{"x": 0, "y": 0}],
                "circle",
                port=0,
                capture_duration=0,
            )

    def test_svgpt_position_query_uses_port_zero_and_quoted_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = Path(temp_dir) / "svgpt_query.html"
            html_path.write_text(SVGPT_QUERY_FIXTURE, encoding="utf-8")

            matches = get_position_in_time(
                str(html_path),
                [{"x": 125, "y": 25}],
                "circle'quoted",
                tolerance=0.01,
                port=0,
            )

        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0])
        best_match = min(matches[0], key=lambda match: match["error"])
        self.assertAlmostEqual(best_match["time"], 0.01, places=6)
        self.assertLessEqual(best_match["error"], 0.01)


if __name__ == "__main__":
    unittest.main()
