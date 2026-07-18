from datetime import datetime

from pydantic import BaseModel

from app.db.models import TaskType


class ProjectCreate(BaseModel):
    name: str
    task_type: TaskType = TaskType.instance_segmentation


class ProjectRead(BaseModel):
    id: int
    name: str
    task_type: TaskType
    created_at: datetime
    image_count: int = 0
    annotated_count: int = 0

    model_config = {"from_attributes": True}


class ProjectUpdate(BaseModel):
    name: str | None = None
