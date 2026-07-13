"""Python-version-stable transform flag helpers."""

from enum import IntFlag


class TransformType(IntFlag):
    NONE = 0

    TRANSLATE = 1 << 0
    ROTATE = 1 << 1
    SCALE = 1 << 2

    ANY = TRANSLATE | ROTATE | SCALE


ATOMIC_TRANSFORM_TYPES = (
    TransformType.TRANSLATE,
    TransformType.ROTATE,
    TransformType.SCALE,
)


def iter_atomic_transform_types(
    transform_types: TransformType,
) -> tuple[TransformType, ...]:
    """Return only atomic flags contained in ``transform_types``."""
    return tuple(
        transform_type
        for transform_type in ATOMIC_TRANSFORM_TYPES
        if transform_type in transform_types
    )
