import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from app.config import settings
from app.core.embedding_cache import EmbeddingCache
from app.core.sam_engine import SAMEngine

logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    """Select the last CUDA device (discrete GPU on dual-GPU laptops), or CPU."""
    if settings.CUDA_DEVICE:
        return settings.CUDA_DEVICE
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.device_count() - 1}"
    logger.warning("CUDA not available — running SAM on CPU (will be slow).")
    return "cpu"


class SAM2Backend(SAMEngine):
    """
    SAM 2.1 backend (variant selected by ``settings.SAM_VARIANT``).

    Embedding cache strategy (three levels, cheapest first):
      1. In-memory:  same image_id as last set_image() call → skip entirely.
      2. Disk cache: .pt file with full feature tensors → restore in ~50 ms.
      3. Fresh:      run forward pass through the image encoder (~1-2 s on RTX 3050).

    Concurrency. A single instance is shared by the whole application and holds
    the selected image in mutable state (``_current_image_id`` and the
    predictor's feature tensors). FastAPI runs sync endpoints in a threadpool,
    so without a lock two overlapping requests interleave set_image() and
    predict(), and a click on one image can be answered with a mask computed
    from another. Every public entry point takes ``_lock``; callers that need
    select-then-predict as one atomic step must use :meth:`predict_on_image`
    instead of calling the two in sequence.
    """

    def __init__(self) -> None:
        self._device = _resolve_device()
        # SAM2ImagePredictor; typed as Any since sam2 ships no type stubs.
        self._predictor: Any = None
        self._current_image_id: int | None = None
        self._cache = EmbeddingCache()
        # Re-entrant: predict_on_image() holds the lock and then calls
        # set_image() and predict_from_points(), which acquire it again.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ckpt = str(settings.sam_checkpoint_path)
        logger.info(
            "Loading SAM 2.1 (%s) on %s  ckpt=%s",
            settings.SAM_VARIANT,
            self._device,
            ckpt,
        )
        if not settings.sam_checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {ckpt}\n"
                f"SAM_VARIANT is '{settings.SAM_VARIANT}', which needs "
                f"'{settings.sam_checkpoint_name}' in {settings.MODELS_DIR}."
            )

        with torch.inference_mode():
            model = build_sam2(settings.sam_config, ckpt, device=self._device)

        # Keep model in fp32 — autocast handles mixed precision during forward pass.
        # model.half() causes dtype mismatch with SAM2's internal float32 preprocessing.
        self._predictor = SAM2ImagePredictor(model)

        # Compile the mask decoder (hot path during predict). The image encoder is
        # compiled separately only when it hasn't been cached yet.
        # mode="reduce-overhead" avoids graph breaks from dynamic shapes in SAM2.
        # torch.compile requires Triton, which is not supported on Windows
        import sys

        if (
            self._device.startswith("cuda")
            and hasattr(torch, "compile")
            and sys.platform != "win32"
        ):
            try:
                self._predictor.model.sam_mask_decoder = torch.compile(
                    self._predictor.model.sam_mask_decoder,
                    mode="reduce-overhead",
                )
                logger.info(
                    "SAM mask decoder compiled with torch.compile (reduce-overhead)."
                )
            except Exception as exc:
                logger.warning("torch.compile() failed (will run interpreted): %s", exc)
        elif sys.platform == "win32":
            logger.info("torch.compile skipped on Windows (Triton not supported).")

        logger.info(
            "SAM 2.1 ready. VRAM allocated: %.0f MB",
            torch.cuda.memory_allocated(self._device) / 1e6,
        )

    def unload_model(self) -> None:
        self._predictor = None
        self._current_image_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM 2.1 unloaded.")

    # ------------------------------------------------------------------
    # Image setup
    # ------------------------------------------------------------------

    def predict_on_image(
        self,
        image_path: Path,
        image_id: int | None,
        project_dir: Path | None,
        points: list[tuple[float, float]],
        labels: list[int],
        box: list[float] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Select an image and predict from it as one atomic operation.

        Prefer this over ``set_image()`` followed by ``predict_from_points()``:
        those take the lock twice, and another request can select a different
        image in between, so the returned mask would be from the wrong image.
        """
        with self._lock:
            self.set_image(image_path, image_id=image_id, project_dir=project_dir)
            return self.predict_from_points(points, labels, box=box)

    def set_image(
        self,
        image_path: Path,
        image_id: int | None = None,
        project_dir: Path | None = None,
    ) -> None:
        with self._lock:
            self._set_image_locked(image_path, image_id, project_dir)

    def _set_image_locked(
        self,
        image_path: Path,
        image_id: int | None,
        project_dir: Path | None,
    ) -> None:
        if self._predictor is None:
            raise RuntimeError("Call load_model() before set_image().")

        # Level 1: in-memory — already set, nothing to do
        if image_id is not None and image_id == self._current_image_id:
            logger.debug("set_image: in-memory hit for image_id=%s", image_id)
            return

        # Level 2: disk cache — restore feature tensors without rerunning encoder
        if image_id is not None:
            cached = self._cache.load(image_id, self._device, project_dir)
            if cached is not None:
                self._restore_features(cached)
                self._current_image_id = image_id
                logger.debug("set_image: disk cache hit for image_id=%s", image_id)
                return

        # Level 3: compute fresh embedding
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            # autocast enables fp16 arithmetic without fp16 model weights
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self._device.startswith("cuda")
                else torch.inference_mode()
            )
            with torch.inference_mode(), ctx:
                self._predictor.set_image(img_rgb)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(
                "OOM during set_image for image_id=%s. "
                "Try closing other GPU applications.",
                image_id,
            )
            raise

        self._current_image_id = image_id

        # Persist to disk so subsequent server restarts skip recomputation
        if image_id is not None and hasattr(self._predictor, "_features"):
            try:
                self._cache.save(
                    image_id,
                    self._predictor._features,
                    self._predictor._orig_hw,
                    project_dir,
                )
            except Exception as exc:
                logger.warning("Could not save embedding cache: %s", exc)

    def _restore_features(self, cached: dict[str, Any]) -> None:
        """Inject cached feature tensors back into the predictor without re-encoding."""
        self._predictor._features = cached["features"]
        self._predictor._orig_hw = cached["orig_hw"]
        self._predictor._is_image_set = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_from_points(
        self,
        points: list[tuple[float, float]],
        labels: list[int],
        box: list[float] | None = None,  # pixel [x1, y1, x2, y2]
    ) -> tuple[np.ndarray, float]:
        with self._lock:
            return self._predict_from_points_locked(points, labels, box)

    def _predict_from_points_locked(
        self,
        points: list[tuple[float, float]],
        labels: list[int],
        box: list[float] | None,
    ) -> tuple[np.ndarray, float]:
        if self._predictor is None:
            raise RuntimeError("Call load_model() first.")
        if self._current_image_id is None:
            raise RuntimeError("Call set_image() before predict_from_points().")

        has_points = len(points) > 0
        has_negative = any(label == 0 for label in labels)
        has_box = box is not None and len(box) == 4

        pts = np.array(points, dtype=np.float32) if has_points else None
        lbl = np.array(labels, dtype=np.int32) if has_points else None
        box_arr = np.array(box, dtype=np.float32) if has_box else None

        # multimask only for a single positive click with no box constraint.
        # Box prompt or negative points → single mask respects all constraints.
        multimask = (not has_box) and (not has_negative) and len(labels) == 1

        try:
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self._device.startswith("cuda")
                else torch.inference_mode()
            )
            with torch.inference_mode(), ctx:
                masks, scores, _ = self._predictor.predict(
                    point_coords=pts,
                    point_labels=lbl,
                    box=box_arr,
                    multimask_output=multimask,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("OOM during predict_from_points. VRAM exhausted.")
            raise

        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])

    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None
