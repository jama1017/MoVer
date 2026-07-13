import unittest

from mover.dsl.transform_types import (
    TransformType,
    iter_atomic_transform_types,
)


class TransformTypeCompatibilityTest(unittest.TestCase):
    def test_iterates_only_requested_atomic_flags(self) -> None:
        self.assertEqual(
            iter_atomic_transform_types(TransformType.TRANSLATE),
            (TransformType.TRANSLATE,),
        )
        self.assertEqual(
            iter_atomic_transform_types(
                TransformType.TRANSLATE | TransformType.ROTATE
            ),
            (TransformType.TRANSLATE, TransformType.ROTATE),
        )

    def test_any_expands_to_atomic_flags_without_yielding_any(self) -> None:
        self.assertEqual(
            iter_atomic_transform_types(TransformType.ANY),
            (
                TransformType.TRANSLATE,
                TransformType.ROTATE,
                TransformType.SCALE,
            ),
        )
        self.assertNotIn(
            TransformType.ANY,
            iter_atomic_transform_types(TransformType.ANY),
        )

    def test_none_has_no_atomic_flags(self) -> None:
        self.assertEqual(
            iter_atomic_transform_types(TransformType.NONE),
            (),
        )


if __name__ == "__main__":
    unittest.main()
