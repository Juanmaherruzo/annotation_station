import json
import random
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import yaml
from sqlmodel import Session, col, select

from app.db.models import Annotation, Image, ImageStatus, LabelClass, Project

_T = TypeVar("_T")


def _bbox_cxcywh(
    points: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """Derive normalized cx, cy, w, h from any list of [x, y] points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


class YOLODetExporter:
    """Export annotations in YOLO detection format (bounding box labels)."""

    def export(
        self,
        project: Project,
        session: Session,
        output_dir: Path,
        splits: dict[str, float] | None = None,
        project_dir: Path | None = None,
    ) -> Path:
        splits = splits or {"train": 0.7, "val": 0.2, "test": 0.1}

        images = list(
            session.exec(
                select(Image).where(
                    Image.project_id == project.id,
                    Image.status == ImageStatus.annotated,
                )
            ).all()
        )

        classes: Sequence[LabelClass] = session.exec(
            select(LabelClass)
            .where(LabelClass.project_id == project.id)
            .order_by(col(LabelClass.yolo_index))
        ).all()

        if not images:
            raise ValueError("No annotated images to export.")

        random.shuffle(images)
        split_images = _make_splits(images, splits)

        export_root = output_dir / f"project_{project.id}_yolo_det"
        if export_root.exists():
            shutil.rmtree(export_root)

        for split_name, split_imgs in split_images.items():
            if not split_imgs:
                continue
            img_dir = export_root / split_name / "images"
            lbl_dir = export_root / split_name / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            for img in split_imgs:
                src = (project_dir or Path()) / "images" / Path(img.filename).name
                if src.exists():
                    shutil.copy2(src, img_dir / Path(img.filename).name)

                annotations: Sequence[Annotation] = session.exec(
                    select(Annotation).where(Annotation.image_id == img.id)
                ).all()

                label_lines = []
                for ann in annotations:
                    cls = session.get(LabelClass, ann.class_id)
                    if cls is None:
                        continue
                    points = json.loads(ann.data)
                    cx, cy, w, h = _bbox_cxcywh(points)
                    label_lines.append(
                        f"{cls.yolo_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    )

                stem = Path(img.filename).stem
                (lbl_dir / f"{stem}.txt").write_text("\n".join(label_lines))

        yaml_data: dict[str, object] = {
            "path": str(export_root),
            "nc": len(classes),
            "names": [cls.name for cls in classes],
        }
        for split_name, split_imgs in split_images.items():
            if split_imgs:
                yaml_data[split_name] = str(export_root / split_name / "images")

        (export_root / "data.yaml").write_text(yaml.dump(yaml_data, allow_unicode=True))

        zip_path = output_dir / f"project_{project.id}_yolo_det.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in export_root.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(output_dir))

        shutil.rmtree(export_root)
        return zip_path


def _make_splits(images: list[_T], ratios: dict[str, float]) -> dict[str, list[_T]]:
    total = len(images)
    result: dict[str, list[_T]] = {}
    cursor = 0
    keys = list(ratios.keys())
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            result[key] = images[cursor:]
        else:
            count = round(ratios[key] * total)
            result[key] = images[cursor : cursor + count]
            cursor += count
    return result
