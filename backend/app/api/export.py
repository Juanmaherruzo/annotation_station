import tempfile
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session

from app.config import settings
from app.core.exporters.coco import COCOExporter
from app.core.exporters.yolo_det import YOLODetExporter
from app.core.exporters.yolo_seg import YOLOSegExporter
from app.db.models import Project
from app.db.session import get_session

router = APIRouter(prefix="/projects/{project_id}/export", tags=["export"])

SessionDep = Annotated[Session, Depends(get_session)]


class Exporter(Protocol):
    """Structural interface shared by every dataset exporter."""

    def export(
        self,
        project: Project,
        session: Session,
        output_dir: Path,
        splits: dict[str, float] | None = None,
        project_dir: Path | None = None,
    ) -> Path: ...


_EXPORTERS: dict[str, type[Exporter]] = {
    "yolo_seg": YOLOSegExporter,
    "yolo_det": YOLODetExporter,
    "coco": COCOExporter,
}


class ExportRequest(BaseModel):
    format: str  # "yolo_seg" | "yolo_det" | "coco"
    splits: dict[str, float] = {"train": 0.7, "val": 0.2, "test": 0.1}


@router.post("/")
def export_project(
    project_id: int, payload: ExportRequest, session: SessionDep
) -> FileResponse:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if payload.format not in _EXPORTERS:
        valid = list(_EXPORTERS)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown format '{payload.format}'. Valid options: {valid}",
        )

    project_dir = settings.DATA_DIR / str(project_id)
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        zip_path = _EXPORTERS[payload.format]().export(
            project=project,
            session=session,
            output_dir=tmp_dir,
            splits=payload.splits,
            project_dir=project_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=zip_path.name,
    )
