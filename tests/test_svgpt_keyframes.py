import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mover.converter.mover_converter import convert_animation


SVGPT_ROOT = Path(__file__).resolve().parents[2] / "SVGPT_shape"
FIXTURE_ROOT = (
    SVGPT_ROOT
    / "dataset"
    / "gt_animations"
    / "star_starol_octagram_shape_params_side_length_50"
)
FIXTURE_STEM = (
    "star_starol_octagram_shape_params_side_length_50"
    "_proto_000_var_000"
)
FIXTURE_HTML = FIXTURE_ROOT / f"{FIXTURE_STEM}.html"
EXPECTED_KEYFRAMES = (
    FIXTURE_ROOT / f"{FIXTURE_STEM}_data_keyframes.json"
)


@unittest.skipUnless(
    FIXTURE_HTML.is_file() and EXPECTED_KEYFRAMES.is_file(),
    "SVGPT_shape compatibility corpus is unavailable",
)
class SvgptTranslationKeyframeCompatibilityTest(unittest.TestCase):
    """Compatibility coverage for the narrow SVGPT translation contract."""

    def test_octagram_translation_keyframes_match_svgpt_fixture(self) -> None:
        expected = json.loads(
            EXPECTED_KEYFRAMES.read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            convert_animation(
                str(FIXTURE_HTML),
                port=0,
                save_keyframes=True,
                output_dir=temp_dir,
            )
            actual_path = (
                Path(temp_dir) / f"{FIXTURE_STEM}_data_keyframes.json"
            )
            actual = json.loads(actual_path.read_text(encoding="utf-8"))

        self.assertEqual(actual["info"], expected["info"])
        self.assertEqual(list(actual)[1], "square")
        self.assertEqual(actual.get("grid"), {})
        self.assertEqual(set(actual["square"]), {"translate"})
        actual_translate = actual["square"]["translate"]
        expected_translate = expected["square"]["translate"]
        self.assertEqual(
            actual_translate["keyframes"],
            expected_translate["keyframes"],
        )
        np.testing.assert_allclose(
            actual_translate["acc_value"],
            expected_translate["acc_value"],
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual_translate["ctm"],
            expected_translate["ctm"],
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual_translate["transformedPts"],
            expected_translate["transformedPts"],
            rtol=0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
