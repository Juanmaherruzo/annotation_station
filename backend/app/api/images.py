import io
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from PIL import Image as PILImage
from sqlmodel import Session, col, select

from app.config import settings
from app.db.models import Annotation, Image, ImageStatus, Project
from app.db.session import get_session
from app.schemas.images import ImageRead, ImageStatusUpdate

router = APIRouter(prefix="/projects/{project_id}/images", tags=["images"])

SessionDep = Annotated[Session, Depends(get_session)]

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _project_dir(project_id: int) -> Path:
    return settings.DATA_DIR / str(project_id)


def _get_image_or_404(image_id: int, project_id: int, session: Session) -> Image:
    img = session.get(Image, image_id)
    if not img or img.project_id != project_id:
        raise HTTPException(status_code=404, detail="Image not found")
    return img


def _get_project_or_404(project_id: int, session: Session) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _safe_filename(name: str, dest_dir: Path) -> str:
    """Append numeric suffix if name already exists in dest_dir."""
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = name
    counter = 1
    while (dest_dir / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


@router.get("/", response_model=list[ImageRead])
def list_images(project_id: int, session: SessionDep) -> Sequence[Image]:
    _get_project_or_404(project_id, session)
    return session.exec(
        select(Image)
        .where(Image.project_id == project_id)
        .order_by(col(Image.created_at))
    ).all()


@router.post("/", response_model=list[ImageRead], status_code=status.HTTP_201_CREATED)
async def upload_images(
    project_id: int,
    session: SessionDep,
    files: list[UploadFile] = File(...),
) -> list[Image]:
    _get_project_or_404(project_id, session)
    base = _project_dir(project_id)
    images_dir = base / "images"
    thumbs_dir = base / "thumbnails"
    images_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    created: list[Image] = []
    for upload in files:
        if upload.filename is None:
            continue  # multipart part without a filename
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            continue  # silently skip unsupported formats

        data = await upload.read()
        pil_img = PILImage.open(io.BytesIO(data)).convert("RGB")
        width, height = pil_img.size

        filename = _safe_filename(upload.filename, images_dir)

        # Save original
        pil_img.save(images_dir / filename)

        # Save thumbnail
        thumb = pil_img.copy()
        thumb.thumbnail(settings.THUMBNAIL_SIZE, PILImage.Resampling.LANCZOS)
        thumb.save(thumbs_dir / filename, "JPEG", quality=80)

        db_image = Image(
            project_id=project_id,
            filename=filename,
            width=width,
            height=height,
            status=ImageStatus.unannotated,
        )
        session.add(db_image)
        session.flush()  # get the id without committing
        created.append(db_image)

    session.commit()
    for img in created:
        session.refresh(img)
    return created


@router.get("/{image_id}/thumbnail")
def get_thumbnail(project_id: int, image_id: int, session: SessionDep) -> Response:
    img = _get_image_or_404(image_id, project_id, session)
    path = _project_dir(project_id) / "thumbnails" / img.filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@router.get("/{image_id}/file")
def get_image_file(project_id: int, image_id: int, session: SessionDep) -> Response:
    img = _get_image_or_404(image_id, project_id, session)
    path = _project_dir(project_id) / "images" / img.filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    suffix = Path(img.filename).suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    return Response(content=path.read_bytes(), media_type=media_type)


@router.patch("/{image_id}/status", response_model=ImageRead)
def update_image_status(
    project_id: int, image_id: int, payload: ImageStatusUpdate, session: SessionDep
) -> Image:
    img = _get_image_or_404(image_id, project_id, session)
    img.status = payload.status
    session.add(img)
    session.commit()
    session.refresh(img)
    return img


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(project_id: int, image_id: int, session: SessionDep) -> None:
    img = _get_image_or_404(image_id, project_id, session)

    # Delete associated annotations
    annotations = session.exec(
        select(Annotation).where(Annotation.image_id == image_id)
    ).all()
    for ann in annotations:
        session.delete(ann)

    session.delete(img)
    session.commit()

    # Remove files from disk
    base = _project_dir(project_id)
    for subdir in ("images", "thumbnails"):
        f = base / subdir / img.filename
        if f.exists():
            f.unlink()

    # Remove cached embedding if present
    emb = base / "_embeddings" / f"{image_id}.npy"
    if emb.exists():
        emb.unlink()
