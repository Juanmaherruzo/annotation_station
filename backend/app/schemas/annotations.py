from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

# Normalized [0,1] coordinate pair
Point = tuple[float, float]


class AnnotationCreate(BaseModel):
    class_id: int
    # Polygon: list of [x,y] pairs, normalized. For detection: [[x,y,w,h]] single entry.
    data: list[Point]


class AnnotationRead(BaseModel):
    id: int
    image_id: int
    class_id: int
    data: list[Point]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AnnotationUpdate(BaseModel):
    data: list[Point] | None = None
    class_id: int | None = None


# ---------------------------------------------------------------------------
# SAM inference request / response
# ---------------------------------------------------------------------------


class InferencePoint(BaseModel):
    x: float  # pixel coordinate
    y: float
    label: Annotated[int, Field(ge=0, le=1)]  # 1=positive, 0=negative


class InferenceRequest(BaseModel):
    image_id: int
    points: list[InferencePoint]
    box: list[float] | None = (
        None  # pixel [x1, y1, x2, y2] — optional SAM region constraint
    )


class InferenceResponse(BaseModel):
    # Normalized polygon vertices [[x,y], ...]
    polygon: list[Point]
    # Bounding box [x, y, w, h] normalized
    bbox: tuple[float, float, float, float]
    score: float
