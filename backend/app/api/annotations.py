import json
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.db.models import Annotation, Image, ImageStatus, LabelClass
from app.db.session import get_session
from app.schemas.annotations import AnnotationCreate, AnnotationRead, AnnotationUpdate

router = APIRouter(
    prefix="/projects/{project_id}/images/{image_id}/annotations",
    tags=["annotations"],
)

SessionDep = Annotated[Session, Depends(get_session)]


def _get_image_or_404(image_id: int, project_id: int, session: Session) -> Image:
    img = session.get(Image, image_id)
    if not img or img.project_id != project_id:
        raise HTTPException(status_code=404, detail="Image not found")
    return img


def _get_annotation_or_404(ann_id: int, image_id: int, session: Session) -> Annotation:
    ann = session.get(Annotation, ann_id)
    if not ann or ann.image_id != image_id:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return ann


def _deserialize(ann: Annotation) -> AnnotationRead:
    assert ann.id is not None  # persisted annotations always carry a primary key
    raw = json.loads(ann.data)
    return AnnotationRead(
        id=ann.id,
        image_id=ann.image_id,
        class_id=ann.class_id,
        data=[tuple(p) for p in raw],
        created_at=ann.created_at,
        updated_at=ann.updated_at,
    )


def _sync_image_status(image: Image, session: Session) -> None:
    """Set image status based on annotation count: annotated ↔ unannotated."""
    annotations = session.exec(
        select(Annotation).where(Annotation.image_id == image.id)
    ).all()
    new_status = ImageStatus.annotated if annotations else ImageStatus.unannotated
    if image.status != new_status:
        image.status = new_status
        session.add(image)


@router.get("/", response_model=list[AnnotationRead])
def list_annotations(
    project_id: int, image_id: int, session: SessionDep
) -> list[AnnotationRead]:
    _get_image_or_404(image_id, project_id, session)
    annotations = session.exec(
        select(Annotation).where(Annotation.image_id == image_id)
    ).all()
    return [_deserialize(a) for a in annotations]


@router.post("/", response_model=AnnotationRead, status_code=status.HTTP_201_CREATED)
def create_annotation(
    project_id: int, image_id: int, payload: AnnotationCreate, session: SessionDep
) -> AnnotationRead:
    image = _get_image_or_404(image_id, project_id, session)

    # Verify class belongs to this project
    cls = session.get(LabelClass, payload.class_id)
    if not cls or cls.project_id != project_id:
        raise HTTPException(status_code=400, detail="Invalid class_id for this project")

    ann = Annotation(
        image_id=image_id,
        class_id=payload.class_id,
        data=json.dumps([list(p) for p in payload.data]),
    )
    session.add(ann)
    _sync_image_status(image, session)
    session.commit()
    session.refresh(ann)
    return _deserialize(ann)


@router.patch("/{annotation_id}", response_model=AnnotationRead)
def update_annotation(
    project_id: int,
    image_id: int,
    annotation_id: int,
    payload: AnnotationUpdate,
    session: SessionDep,
) -> AnnotationRead:
    _get_image_or_404(image_id, project_id, session)
    ann = _get_annotation_or_404(annotation_id, image_id, session)

    if payload.data is not None:
        ann.data = json.dumps([list(p) for p in payload.data])
    if payload.class_id is not None:
        cls = session.get(LabelClass, payload.class_id)
        if not cls or cls.project_id != project_id:
            raise HTTPException(
                status_code=400, detail="Invalid class_id for this project"
            )
        ann.class_id = payload.class_id
    ann.updated_at = datetime.utcnow()

    session.add(ann)
    session.commit()
    session.refresh(ann)
    return _deserialize(ann)


@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(
    project_id: int, image_id: int, annotation_id: int, session: SessionDep
) -> None:
    image = _get_image_or_404(image_id, project_id, session)
    ann = _get_annotation_or_404(annotation_id, image_id, session)
    session.delete(ann)
    session.flush()
    _sync_image_status(image, session)
    session.commit()
