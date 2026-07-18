import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from app.config import settings


def mask_to_polygon(
    mask: np.ndarray,
    tolerance: float | None = None,
) -> list[tuple[float, float]]:
    """
    Convert a binary mask [H, W] to a simplified polygon in pixel coordinates.
    Uses OpenCV contour extraction + Shapely simplification.
    Returns the largest contour as a list of (x, y) tuples.
    """
    import cv2

    tol = tolerance if tolerance is not None else settings.POLYGON_TOLERANCE
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).tolist()
    if len(pts) < 3:
        return [(float(p[0]), float(p[1])) for p in pts]

    poly = Polygon(pts)
    poly = make_valid(poly)
    if poly.is_empty:
        return []

    # make_valid may return MultiPolygon on self-intersection; keep the largest part
    if poly.geom_type != "Polygon":
        parts = list(poly.geoms) if hasattr(poly, "geoms") else []
        if not parts:
            return []
        poly = max(parts, key=lambda g: g.area)

    simplified = poly.simplify(tol, preserve_topology=True)

    # simplify with preserve_topology can still return MultiPolygon in edge cases
    if simplified.geom_type != "Polygon":
        parts = list(simplified.geoms) if hasattr(simplified, "geoms") else []
        if not parts:
            return []
        simplified = max(parts, key=lambda g: g.area)

    coords = list(simplified.exterior.coords)
    return [(float(x), float(y)) for x, y in coords]


def polygon_to_normalized(
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Scale pixel coordinates to [0, 1] range."""
    return [(x / width, y / height) for x, y in polygon]


def normalized_to_pixel(
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Scale normalized [0, 1] coordinates back to pixel space."""
    return [(x * width, y * height) for x, y in polygon]


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return bounding box [x, y, w, h] in pixel coordinates from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)


def bbox_to_normalized(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Normalize a pixel [x, y, w, h] bbox to [0, 1] range."""
    x, y, w, h = bbox
    return x / width, y / height, w / width, h / height
