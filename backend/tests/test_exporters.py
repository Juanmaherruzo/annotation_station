"""End-to-end tests for the export classes using an in-memory SQLite database.

Each exporter is exercised against a real (tiny) dataset: one annotated image
with a single square polygon. Image files are absent, so only labels/metadata
are written — which is exactly the logic under test.
"""

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.core.exporters.coco import COCOExporter, _polygon_area
from app.core.exporters.yolo_det import YOLODetExporter, _bbox_cxcywh
from app.core.exporters.yolo_seg import YOLOSegExporter, _make_splits
from app.db.models import Annotation, Image, ImageStatus, LabelClass, Project

# Normalized square polygon (fractions of image size).
_SQUARE = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        yield db


@pytest.fixture
def project(session: Session) -> Project:
    project = Project(name="test-project")
    session.add(project)
    session.commit()
    session.refresh(project)

    label = LabelClass(project_id=project.id, name="tree", yolo_index=0)
    session.add(label)
    session.commit()
    session.refresh(label)

    image = Image(
        project_id=project.id,
        filename="img_001.jpg",
        width=100,
        height=100,
        status=ImageStatus.annotated,
    )
    session.add(image)
    session.commit()
    session.refresh(image)

    session.add(
        Annotation(image_id=image.id, class_id=label.id, data=json.dumps(_SQUARE))
    )
    session.commit()
    return project


def _unzip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def test_coco_export_produces_valid_json(
    session: Session, project: Project, tmp_path: Path
) -> None:
    zip_path = COCOExporter().export(project, session, tmp_path)
    assert zip_path.exists()

    out = tmp_path / "unzipped"
    _unzip(zip_path, out)
    root = out / f"project_{project.id}_coco"
    coco = json.loads((root / "train" / "instances_train.json").read_text())

    assert coco["categories"] == [{"id": 0, "name": "tree", "supercategory": ""}]
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    assert ann["bbox"] == [10.0, 10.0, 40.0, 40.0]
    assert ann["area"] == 1600.0


def test_yolo_seg_export_writes_polygon_labels(
    session: Session, project: Project, tmp_path: Path
) -> None:
    zip_path = YOLOSegExporter().export(project, session, tmp_path)
    out = tmp_path / "unzipped"
    _unzip(zip_path, out)
    root = out / f"project_{project.id}_yolo_seg"

    label = (root / "train" / "labels" / "img_001.txt").read_text().strip()
    assert label.startswith("0 ")  # class index
    assert len(label.split()) == 1 + 2 * len(_SQUARE)  # index + x/y per vertex

    data_yaml = (root / "data.yaml").read_text()
    assert "nc: 1" in data_yaml
    assert "tree" in data_yaml


def test_yolo_det_export_writes_bbox_labels(
    session: Session, project: Project, tmp_path: Path
) -> None:
    zip_path = YOLODetExporter().export(project, session, tmp_path)
    out = tmp_path / "unzipped"
    _unzip(zip_path, out)
    root = out / f"project_{project.id}_yolo_det"

    label = (root / "train" / "labels" / "img_001.txt").read_text().strip()
    parts = label.split()
    assert parts[0] == "0"
    assert len(parts) == 5  # class cx cy w h


def test_export_without_annotated_images_raises(
    session: Session, tmp_path: Path
) -> None:
    empty = Project(name="empty")
    session.add(empty)
    session.commit()
    session.refresh(empty)
    with pytest.raises(ValueError, match="No annotated images"):
        COCOExporter().export(empty, session, tmp_path)


def test_polygon_area_of_unit_square() -> None:
    assert _polygon_area([[0, 0], [4, 0], [4, 4], [0, 4]]) == 16.0


def test_bbox_cxcywh_of_square() -> None:
    assert _bbox_cxcywh([[0.1, 0.1], [0.5, 0.5]]) == (0.3, 0.3, 0.4, 0.4)


def test_make_splits_last_key_gets_remainder() -> None:
    items = list(range(10))
    splits = _make_splits(items, {"train": 0.7, "val": 0.2, "test": 0.1})
    assert len(splits["train"]) == 7
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 1
    assert splits["train"] + splits["val"] + splits["test"] == items
