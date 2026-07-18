from datetime import datetime

from pydantic import BaseModel

from app.db.models import ImageStatus


class ImageRead(BaseModel):
    id: int
    project_id: int
    filename: str
    width: int
    height: int
    status: ImageStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class ImageStatusUpdate(BaseModel):
    status: ImageStatus
