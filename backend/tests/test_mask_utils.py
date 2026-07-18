"""Unit tests for the pure geometry helpers in ``app.core.mask_utils``."""

import numpy as np

from app.core import mask_utils


def test_mask_to_bbox_returns_tight_box() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True  # rows 2-4, cols 3-6
    # bbox = (x, y, w, h) with w/h measured as (max - min) of set pixels
    assert mask_utils.mask_to_bbox(mask) == (3, 2, 3, 2)


def test_mask_to_bbox_empty_mask_is_zero() -> None:
    assert mask_utils.mask_to_bbox(np.zeros((5, 5), dtype=bool)) == (0, 0, 0, 0)


def test_polygon_normalize_roundtrip() -> None:
    poly = [(10.0, 20.0), (30.0, 40.0)]
    norm = mask_utils.polygon_to_normalized(poly, width=100, height=200)
    assert norm == [(0.1, 0.1), (0.3, 0.2)]
    assert mask_utils.normalized_to_pixel(norm, width=100, height=200) == poly


def test_bbox_to_normalized() -> None:
    assert mask_utils.bbox_to_normalized((10, 20, 30, 40), 100, 200) == (
        0.1,
        0.1,
        0.3,
        0.2,
    )


def test_mask_to_polygon_of_square_has_enough_vertices() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    poly = mask_utils.mask_to_polygon(mask, tolerance=0.5)
    assert len(poly) >= 4
    assert all(len(pt) == 2 for pt in poly)


def test_mask_to_polygon_empty_mask_is_empty() -> None:
    assert mask_utils.mask_to_polygon(np.zeros((10, 10), dtype=bool)) == []
