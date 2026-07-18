import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session

from app.config import settings
from app.core.mask_utils import (
    bbox_to_normalized,
    mask_to_bbox,
    mask_to_polygon,
    polygon_to_normalized,
)
from app.core.sam2_backend import SAM2Backend
from app.db.models import Image
from app.db.session import get_session
from app.schemas.annotations import InferenceRequest, InferenceResponse

router = APIRouter(prefix="/inference", tags=["inference"])

SessionDep = Annotated[Session, Depends(get_session)]

logger = logging.getLogger(__name__)


def _get_engine(request: Request) -> SAM2Backend:
    engine: SAM2Backend = request.app.state.sam_engine
    return engine


@router.post("/precompute", status_code=202)
def precompute_embedding(
    payload: dict[str, int],
    request: Request,
    session: SessionDep,
) -> dict[str, object]:
    """
    Precompute and cache the image embedding without running prediction.
    Call this as soon as the user selects an image so the first SAM click is instant.
    """
    image_id = payload.get("image_id")
    if not image_id:
        raise HTTPException(status_code=422, detail="image_id required")

    image = session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = settings.DATA_DIR / str(image.project_id) / "images" / image.filename
    project_dir = settings.DATA_DIR / str(image.project_id)

    engine: SAM2Backend = _get_engine(request)
    try:
        engine.set_image(image_path, image_id=image.id, project_dir=project_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    except MemoryError:
        raise HTTPException(status_code=507, detail="Insufficient VRAM")

    logger.info("inference/precompute: image_id=%s cached", image_id)
    return {"status": "ok", "image_id": image_id}


@router.post("/point", response_model=InferenceResponse)
def predict_from_points(
    payload: InferenceRequest,
    request: Request,
    session: SessionDep,
) -> InferenceResponse:
    """
    Receive click prompts, run SAM 2.1, return the best mask as a normalized polygon.

    Flow:
      1. Look up image in DB to resolve project folder and filename.
      2. Call set_image() — returns immediately on cache hit (~0 ms), otherwise
         runs the image encoder (~1-2 s on RTX 3050) and caches the result.
      3. Call predict_from_points() using cached features (~100-300 ms).
      4. Convert binary mask → simplified polygon → normalize to [0, 1].
    """
    image = session.get(Image, payload.image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = settings.DATA_DIR / str(image.project_id) / "images" / image.filename
    project_dir = settings.DATA_DIR / str(image.project_id)

    engine: SAM2Backend = _get_engine(request)

    # Compute or restore image embedding
    try:
        engine.set_image(image_path, image_id=image.id, project_dir=project_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    except MemoryError:
        raise HTTPException(status_code=507, detail="Insufficient VRAM for this image")

    # Build point arrays in pixel space
    points = [(pt.x, pt.y) for pt in payload.points]
    labels = [pt.label for pt in payload.points]

    if not points:
        raise HTTPException(status_code=422, detail="At least one point is required")

    # Denormalize box from normalized [[x1,y1],[x2,y2]] to pixel [x1,y1,x2,y2]
    box_pixels = None
    if payload.box and len(payload.box) == 4:
        x1, y1, x2, y2 = payload.box
        box_pixels = [
            x1 * image.width,
            y1 * image.height,
            x2 * image.width,
            y2 * image.height,
        ]

    try:
        mask, score = engine.predict_from_points(points, labels, box=box_pixels)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not mask.any():
        raise HTTPException(status_code=422, detail="SAM returned an empty mask")

    # mask → pixel polygon → normalized polygon
    pixel_polygon = mask_to_polygon(mask)
    if not pixel_polygon:
        raise HTTPException(
            status_code=422, detail="Could not extract polygon from mask"
        )

    norm_polygon = polygon_to_normalized(pixel_polygon, image.width, image.height)

    # Bounding box from mask
    pixel_bbox = mask_to_bbox(mask)
    norm_bbox = bbox_to_normalized(pixel_bbox, image.width, image.height)

    logger.info(
        "inference/point: image_id=%s  points=%d  score=%.3f  vertices=%d",
        payload.image_id,
        len(points),
        score,
        len(norm_polygon),
    )

    return InferenceResponse(polygon=norm_polygon, bbox=norm_bbox, score=score)
