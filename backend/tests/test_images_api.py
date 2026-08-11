"""HTTP-level tests for the images endpoints.

The suite previously had no endpoint test at all, despite httpx being declared
as a dev dependency. These cover the upload path, where two defects lived:

- the uploaded filename was not stripped of directory components, so a part
  named ``../../../evil.png`` was written outside the project directory;
- unsupported or corrupt files were skipped silently and the response gave the
  caller no way to know which ones did not make it.

The app under test is assembled here from the projects and images routers alone,
rather than imported from ``app.main``. ``app.main`` pulls in the SAM backend and
therefore ``torch``, which is an optional extra (``.[inference]``); building the
router directly keeps these tests runnable on CI with only ``.[dev]`` installed,
with no GPU and no checkpoint.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    data_dir = tmp_path / "projects"

    from app.api import images as images_api
    from app.api import projects as projects_api
    from app.db.session import get_session

    monkeypatch.setattr(images_api.settings, "DATA_DIR", data_dir)

    app = FastAPI()
    app.include_router(projects_api.router, prefix="/api")
    app.include_router(images_api.router, prefix="/api")

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def _session() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = _session
    yield TestClient(app)
    app.dependency_overrides.clear()


def _png_bytes(size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, (10, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_project(client: TestClient, name: str = "test") -> int:
    response = client.post("/api/projects/", json={"name": name})
    assert response.status_code in (200, 201), response.text
    project_id: int = response.json()["id"]
    return project_id


def test_upload_accepts_a_png(client: TestClient) -> None:
    project_id = _make_project(client)
    response = client.post(
        f"/api/projects/{project_id}/images/",
        files=[("files", ("tree.png", _png_bytes(), "image/png"))],
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["created"]) == 1
    assert body["created"][0]["filename"] == "tree.png"
    assert body["created"][0]["width"] == 32
    assert body["skipped"] == []


def test_upload_strips_directory_traversal_from_the_filename(
    client: TestClient, tmp_path: Path
) -> None:
    """A crafted filename must not escape the project directory."""
    project_id = _make_project(client)
    response = client.post(
        f"/api/projects/{project_id}/images/",
        files=[("files", ("../../../escaped.png", _png_bytes(), "image/png"))],
    )
    assert response.status_code == 201, response.text
    body = response.json()

    assert len(body["created"]) == 1
    assert body["created"][0]["filename"] == "escaped.png"

    images_dir = tmp_path / "projects" / str(project_id) / "images"
    assert (images_dir / "escaped.png").is_file()
    # Nothing was written above the project tree.
    assert not (tmp_path / "escaped.png").exists()
    assert not (tmp_path.parent / "escaped.png").exists()


def test_unsupported_formats_are_reported_not_silently_dropped(
    client: TestClient,
) -> None:
    project_id = _make_project(client)
    response = client.post(
        f"/api/projects/{project_id}/images/",
        files=[
            ("files", ("good.png", _png_bytes(), "image/png")),
            ("files", ("scan.tif", b"II*\x00fake tiff", "image/tiff")),
        ],
    )
    assert response.status_code == 201, response.text
    body = response.json()

    assert [img["filename"] for img in body["created"]] == ["good.png"]
    assert len(body["skipped"]) == 1
    assert body["skipped"][0]["name"] == "scan.tif"
    assert ".tif" in body["skipped"][0]["reason"]


def test_a_corrupt_file_does_not_abort_the_batch(client: TestClient) -> None:
    """One unreadable image must not cost the caller the whole upload."""
    project_id = _make_project(client)
    response = client.post(
        f"/api/projects/{project_id}/images/",
        files=[
            ("files", ("broken.png", b"not actually a png", "image/png")),
            ("files", ("fine.png", _png_bytes(), "image/png")),
        ],
    )
    assert response.status_code == 201, response.text
    body = response.json()

    assert [img["filename"] for img in body["created"]] == ["fine.png"]
    assert [s["name"] for s in body["skipped"]] == ["broken.png"]


def test_colliding_filenames_are_de_duplicated(client: TestClient) -> None:
    project_id = _make_project(client)
    for _ in range(2):
        client.post(
            f"/api/projects/{project_id}/images/",
            files=[("files", ("same.png", _png_bytes(), "image/png"))],
        )
    listing = client.get(f"/api/projects/{project_id}/images/")
    assert listing.status_code == 200, listing.text
    names = sorted(img["filename"] for img in listing.json())
    assert names == ["same.png", "same_1.png"]


def test_deleting_an_image_removes_its_cached_embedding(
    client: TestClient, tmp_path: Path
) -> None:
    """The delete path looked for a .npy while the cache writes .pt.

    Nothing was ever removed, so embeddings accumulated indefinitely.
    """
    project_id = _make_project(client)
    upload = client.post(
        f"/api/projects/{project_id}/images/",
        files=[("files", ("tree.png", _png_bytes(), "image/png"))],
    )
    image_id = upload.json()["created"][0]["id"]

    project_dir = tmp_path / "projects" / str(project_id)
    embeddings = project_dir / "_embeddings"
    embeddings.mkdir(parents=True, exist_ok=True)
    cached = embeddings / f"{image_id}.pt"
    cached.write_bytes(b"stand-in for a feature tensor")

    response = client.delete(f"/api/projects/{project_id}/images/{image_id}")
    assert response.status_code == 204, response.text
    assert not cached.exists()
    assert not (project_dir / "images" / "tree.png").exists()
