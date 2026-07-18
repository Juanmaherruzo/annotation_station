from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, col, select

from app.db.models import LabelClass, Project
from app.db.session import get_session
from app.schemas.classes import ClassCreate, ClassRead, ClassReorderRequest, ClassUpdate

router = APIRouter(prefix="/projects/{project_id}/classes", tags=["classes"])

SessionDep = Annotated[Session, Depends(get_session)]


def _get_project_or_404(project_id: int, session: Session) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _get_class_or_404(class_id: int, project_id: int, session: Session) -> LabelClass:
    cls = session.get(LabelClass, class_id)
    if not cls or cls.project_id != project_id:
        raise HTTPException(status_code=404, detail="Class not found")
    return cls


@router.get("/", response_model=list[ClassRead])
def list_classes(project_id: int, session: SessionDep) -> Sequence[LabelClass]:
    _get_project_or_404(project_id, session)
    classes = session.exec(
        select(LabelClass)
        .where(LabelClass.project_id == project_id)
        .order_by(col(LabelClass.yolo_index))
    ).all()
    return classes


@router.post("/", response_model=ClassRead, status_code=status.HTTP_201_CREATED)
def create_class(
    project_id: int, payload: ClassCreate, session: SessionDep
) -> LabelClass:
    _get_project_or_404(project_id, session)

    # Auto-assign the next YOLO index
    existing = session.exec(
        select(LabelClass).where(LabelClass.project_id == project_id)
    ).all()
    next_index = max((c.yolo_index for c in existing), default=-1) + 1

    cls = LabelClass(
        project_id=project_id,
        name=payload.name,
        color=payload.color,
        yolo_index=next_index,
    )
    session.add(cls)
    session.commit()
    session.refresh(cls)
    return cls


@router.patch("/reorder", response_model=list[ClassRead])
def reorder_classes(
    project_id: int, payload: ClassReorderRequest, session: SessionDep
) -> Sequence[LabelClass]:
    """Assign new yolo_index values to all classes in a single atomic operation."""
    _get_project_or_404(project_id, session)
    for item in payload.order:
        cls = session.get(LabelClass, item.id)
        if cls and cls.project_id == project_id:
            cls.yolo_index = item.yolo_index
            session.add(cls)
    session.commit()
    return session.exec(
        select(LabelClass)
        .where(LabelClass.project_id == project_id)
        .order_by(col(LabelClass.yolo_index))
    ).all()


@router.patch("/{class_id}", response_model=ClassRead)
def update_class(
    project_id: int, class_id: int, payload: ClassUpdate, session: SessionDep
) -> LabelClass:
    cls = _get_class_or_404(class_id, project_id, session)
    if payload.name is not None:
        cls.name = payload.name
    if payload.color is not None:
        cls.color = payload.color
    if payload.yolo_index is not None:
        cls.yolo_index = payload.yolo_index
    session.add(cls)
    session.commit()
    session.refresh(cls)
    return cls


@router.delete("/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_class(project_id: int, class_id: int, session: SessionDep) -> None:
    cls = _get_class_or_404(class_id, project_id, session)

    # Delete all annotations that reference this class
    for ann in cls.annotations:
        session.delete(ann)
    session.delete(cls)
    session.commit()

    # Re-index remaining classes to keep yolo_index contiguous
    remaining = session.exec(
        select(LabelClass)
        .where(LabelClass.project_id == project_id)
        .order_by(col(LabelClass.yolo_index))
    ).all()
    for i, c in enumerate(remaining):
        c.yolo_index = i
        session.add(c)
    session.commit()
