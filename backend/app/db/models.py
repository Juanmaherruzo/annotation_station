from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel


class TaskType(str, Enum):
    instance_segmentation = "instance_segmentation"
    object_detection = "object_detection"


class ImageStatus(str, Enum):
    unannotated = "unannotated"
    in_progress = "in_progress"
    annotated = "annotated"


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    task_type: TaskType = Field(default=TaskType.instance_segmentation)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    classes: list["LabelClass"] = Relationship(back_populates="project")
    images: list["Image"] = Relationship(back_populates="project")


# ---------------------------------------------------------------------------
# LabelClass
# ---------------------------------------------------------------------------


class LabelClass(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    name: str
    color: str = Field(default="#FF0000")  # hex color string
    yolo_index: int  # 0-based, auto-assigned on creation

    project: Optional[Project] = Relationship(back_populates="classes")
    annotations: list["Annotation"] = Relationship(back_populates="label_class")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    filename: str  # original filename, stored under project dir
    width: int
    height: int
    status: ImageStatus = Field(default=ImageStatus.unannotated)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    project: Optional[Project] = Relationship(back_populates="images")
    annotations: list["Annotation"] = Relationship(back_populates="image")


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


class Annotation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="image.id")
    class_id: int = Field(foreign_key="labelclass.id")
    # Normalized polygon [[x,y], ...] for segmentation, or [x,y,w,h] for detection
    data: str  # JSON-serialized list
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    image: Optional[Image] = Relationship(back_populates="annotations")
    label_class: Optional[LabelClass] = Relationship(back_populates="annotations")
