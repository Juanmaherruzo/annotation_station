import json
import random
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path

import yaml
from sqlmodel import Session, col, select

from app.db.models import Annotation, Image, ImageStatus, LabelClass, Project


class YOLOSegExporter:
    """Export annotations in YOLO segmentation format (polygon labels)."""

    def export(
        self,
        project: Project,
        session: Session,
        output_dir: Path,
        splits: dict[str, float] | None = None,
        project_dir: Path | None = None,
    ) -> Path:
        splits = splits or {"train": 0.7, "val": 0.2, "test": 0.1}

        # Fetch annotated images and their label classes
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

        # Shuffle and split
        random.shuffle(images)
        split_images = _make_splits(images, splits)

        export_root = output_dir / f"project_{project.id}_yolo_seg"
        if export_root.exists():
            shutil.rmtree(export_root)

        # Write images + labels per split
        for split_name, split_imgs in split_images.items():
            if not split_imgs:
                continue
            img_dir = export_root / split_name / "images"
            lbl_dir = export_root / split_name / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            for img in split_imgs:
                # Copy image file
                src = (project_dir or Path()) / "images" / Path(img.filename).name
                if src.exists():
                    shutil.copy2(src, img_dir / Path(img.filename).name)

                # Build label file
                annotations: Sequence[Annotation] = session.exec(
                    select(Annotation).where(Annotation.image_id == img.id)
                ).all()

                label_lines = []
                for ann in annotations:
                    cls = session.get(LabelClass, ann.class_id)
                    if cls is None:
                        continue
                    points = json.loads(ann.data)  # [[x, y], ...]
                    flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
                    label_lines.append(f"{cls.yolo_index} {flat}")

                stem = Path(img.filename).stem
                (lbl_dir / f"{stem}.txt").write_text("\n".join(label_lines))

        # data.yaml
        yaml_data = {
            "path": str(export_root),
            "nc": len(classes),
            "names": [cls.name for cls in classes],
        }
        for split_name in split_images:
            if split_images[split_name]:
                yaml_data[split_name] = str(export_root / split_name / "images")

        (export_root / "data.yaml").write_text(yaml.dump(yaml_data, allow_unicode=True))

        # Zip
        zip_path = output_dir / f"project_{project.id}_yolo_seg.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in export_root.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(output_dir))

        shutil.rmtree(export_root)
        return zip_path


def _make_splits(
    images: list[Image], ratios: dict[str, float]
) -> dict[str, list[Image]]:
    total = len(images)
    result: dict[str, list[Image]] = {}
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
