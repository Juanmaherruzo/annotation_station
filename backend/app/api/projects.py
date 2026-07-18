import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, col, select

from app.config import settings
from app.db.models import Image, ImageStatus, Project
from app.db.session import get_session
from app.schemas.projects import ProjectCreate, ProjectRead, ProjectUpdate

router = APIRouter(prefix="/projects", tags=["projects"])

SessionDep = Annotated[Session, Depends(get_session)]


def _project_dir(project_id: int) -> Path:
    return settings.DATA_DIR / str(project_id)


def _get_or_404(project_id: int, session: Session) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _enrich(project: Project, session: Session) -> ProjectRead:
    images = session.exec(select(Image).where(Image.project_id == project.id)).all()
    annotated = sum(1 for img in images if img.status == ImageStatus.annotated)
    return ProjectRead(
        **project.model_dump(),
        image_count=len(images),
        annotated_count=annotated,
    )


@router.get("/", response_model=list[ProjectRead])
def list_projects(session: SessionDep) -> list[ProjectRead]:
    projects = session.exec(
        select(Project).order_by(col(Project.created_at).desc())
    ).all()
    return [_enrich(p, session) for p in projects]


@router.post("/", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate, session: SessionDep) -> ProjectRead:
    project = Project(name=payload.name, task_type=payload.task_type)
    session.add(project)
    session.commit()
    session.refresh(project)

    # Create project directory layout
    assert project.id is not None  # set by the DB on commit
    base = _project_dir(project.id)
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "thumbnails").mkdir(parents=True, exist_ok=True)
    (base / "_embeddings").mkdir(parents=True, exist_ok=True)

    return _enrich(project, session)


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: int, session: SessionDep) -> ProjectRead:
    return _enrich(_get_or_404(project_id, session), session)


@router.patch("/{project_id}", response_model=ProjectRead)
def update_project(
    project_id: int, payload: ProjectUpdate, session: SessionDep
) -> ProjectRead:
    project = _get_or_404(project_id, session)
    if payload.name is not None:
        project.name = payload.name
    session.add(project)
    session.commit()
    session.refresh(project)
    return _enrich(project, session)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, session: SessionDep) -> None:
    project = _get_or_404(project_id, session)

    # Remove all DB rows (annotations, images, classes) via cascaded deletes in memory
    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    for img in images:
        for ann in img.annotations:
            session.delete(ann)
        session.delete(img)
    for cls in project.classes:
        session.delete(cls)
    session.delete(project)
    session.commit()

    # Remove files on disk
    project_dir = _project_dir(project_id)
    if project_dir.exists():
        shutil.rmtree(project_dir)
