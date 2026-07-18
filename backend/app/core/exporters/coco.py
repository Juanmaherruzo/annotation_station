import json
import random
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from sqlmodel import Session, col, select

from app.db.models import Annotation, Image, ImageStatus, LabelClass, Project


class COCOExporter:
    """Export annotations in COCO JSON format with polygon segmentation."""

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

        export_root = output_dir / f"project_{project.id}_coco"
        if export_root.exists():
            shutil.rmtree(export_root)

        categories = [
            {"id": cls.yolo_index, "name": cls.name, "supercategory": ""}
            for cls in classes
        ]

        for split_name, split_imgs in split_images.items():
            if not split_imgs:
                continue

            img_dir = export_root / split_name / "images"
            img_dir.mkdir(parents=True)

            coco_images = []
            coco_annotations = []
            ann_id = 1

            for img in split_imgs:
                coco_images.append(
                    {
                        "id": img.id,
                        "file_name": Path(img.filename).name,
                        "width": img.width,
                        "height": img.height,
                    }
                )

                src = (project_dir or Path()) / "images" / Path(img.filename).name
                if src.exists():
                    shutil.copy2(src, img_dir / Path(img.filename).name)

                annotations: Sequence[Annotation] = session.exec(
                    select(Annotation).where(Annotation.image_id == img.id)
                ).all()

                for ann in annotations:
                    cls = session.get(LabelClass, ann.class_id)
                    if cls is None:
                        continue

                    norm_pts = json.loads(ann.data)
                    px_pts = [[x * img.width, y * img.height] for x, y in norm_pts]

                    xs = [p[0] for p in px_pts]
                    ys = [p[1] for p in px_pts]
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                    bbox_w, bbox_h = x2 - x1, y2 - y1

                    # Expand 2-point box to 4-corner polygon for segmentation field
                    if len(px_pts) == 2:
                        area = bbox_w * bbox_h
                        px_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    else:
                        area = _polygon_area(px_pts)

                    segmentation = [coord for pt in px_pts for coord in pt]

                    coco_annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img.id,
                            "category_id": cls.yolo_index,
                            "segmentation": [segmentation],
                            "bbox": [
                                round(x1, 2),
                                round(y1, 2),
                                round(bbox_w, 2),
                                round(bbox_h, 2),
                            ],
                            "area": round(area, 2),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

            coco_data = {
                "info": {"description": project.name, "version": "1.0"},
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": categories,
            }
            ann_file = export_root / split_name / f"instances_{split_name}.json"
            ann_file.parent.mkdir(parents=True, exist_ok=True)
            ann_file.write_text(json.dumps(coco_data, indent=2))

        zip_path = output_dir / f"project_{project.id}_coco.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in export_root.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(output_dir))

        shutil.rmtree(export_root)
        return zip_path


_T = TypeVar("_T")


def _polygon_area(pts: Sequence[Sequence[float]]) -> float:
    """Shoelace formula for polygon area in pixel² ."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2


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
