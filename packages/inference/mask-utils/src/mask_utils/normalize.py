"""坐标归一化。"""

from __future__ import annotations

# 统一保留 6 位小数，与平台 BboxAnnotation / PolygonAnnotation 的 value 字段精度对齐。
_NORM_PRECISION = 6


def normalize_coords(
    coords: list[list[float]], width: int, height: int
) -> list[list[float]]:
    """像素坐标 -> 归一化 [0, 1]，6 位小数对齐协议精度。"""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    return [
        [round(x / width, _NORM_PRECISION), round(y / height, _NORM_PRECISION)]
        for x, y in coords
    ]
