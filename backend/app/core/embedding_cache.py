import logging
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Persist SAM2 image feature tensors to disk as .pt files.

    Saves the full predictor feature state (image_embed + high_res_feats + orig_hw)
    so the server can restore a previous embedding after restart without recomputing.
    Tensors are stored on CPU (fp16) and moved to the target device on load.

    ``torch`` is imported lazily inside :meth:`save` and :meth:`load` rather than
    at module scope. The path convention lives here and the images API calls
    :meth:`delete` to clean up after an image, so importing this module must not
    drag the whole ML stack into a request path that only touches the filesystem
    — torch is an optional extra (``.[inference]``), not a core dependency.
    """

    def _path(self, image_id: int, project_dir: Path | None) -> Path:
        base = project_dir or settings.DATA_DIR
        cache_dir = base / "_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{image_id}.pt"

    def exists(self, image_id: int, project_dir: Path | None = None) -> bool:
        return self._path(image_id, project_dir).exists()

    def save(
        self,
        image_id: int,
        features: dict[str, Any],  # predictor._features
        orig_hw: list[int],  # predictor._orig_hw
        project_dir: Path | None = None,
    ) -> None:
        import torch

        path = self._path(image_id, project_dir)

        # Move tensors to CPU and cast to fp16 to halve disk size
        def _to_cpu_fp16(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().to(torch.float16)
            if isinstance(obj, list):
                return [_to_cpu_fp16(x) for x in obj]
            return obj

        payload = {
            "features": {k: _to_cpu_fp16(v) for k, v in features.items()},
            "orig_hw": orig_hw,
            "model": settings.sam_checkpoint_name,  # invalidate on model change
        }
        torch.save(payload, path)
        logger.debug("Embedding saved: %s (%.1f MB)", path, path.stat().st_size / 1e6)

    def load(
        self,
        image_id: int,
        device: str,
        project_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        """
        Return {"features": ..., "orig_hw": ...} with tensors on `device` in fp32,
        or None if no cache file exists.
        """
        import torch

        path = self._path(image_id, project_dir)
        if not path.exists():
            return None

        # weights_only=True refuses to execute arbitrary pickle opcodes. The
        # payload is only tensors, a list and a string, so the safe loader is
        # sufficient; the permissive one was never needed here.
        try:
            payload: dict[str, Any] = torch.load(
                path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            # Written by an older build, or a truncated write. This is derived
            # data: drop it and recompute rather than fail the request.
            logger.warning(
                "Discarding unreadable embedding cache %s: %s", path.name, exc
            )
            path.unlink(missing_ok=True)
            return None

        # Discard cache if it was built with a different model checkpoint
        if payload.get("model") != settings.sam_checkpoint_name:
            logger.warning(
                "Embedding cache for image_id=%s built with '%s', "
                "current model is '%s' — discarding.",
                image_id,
                payload.get("model"),
                settings.sam_checkpoint_name,
            )
            path.unlink(missing_ok=True)
            return None

        def _to_device_fp32(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.to(device=device, dtype=torch.float32)
            if isinstance(obj, list):
                return [_to_device_fp32(x) for x in obj]
            return obj

        payload["features"] = {
            k: _to_device_fp32(v) for k, v in payload["features"].items()
        }
        logger.debug("Embedding restored from cache: image_id=%s", image_id)
        return payload

    def delete(self, image_id: int, project_dir: Path | None = None) -> None:
        path = self._path(image_id, project_dir)
        if path.exists():
            path.unlink()
            logger.debug("Embedding deleted: %s", path)
