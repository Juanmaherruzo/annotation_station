from abc import ABC, abstractmethod
from pathlib import Path


class SAMEngine(ABC):
    """Abstract interface for SAM backends (SAM 2.1, future SAM 3, etc.)."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights onto the configured device."""

    @abstractmethod
    def unload_model(self) -> None:
        """Release model from GPU memory and call torch.cuda.empty_cache()."""

    @abstractmethod
    def set_image(
        self,
        image_path: Path,
        image_id: int | None = None,
        project_dir: Path | None = None,
    ) -> None:
        """
        Prepare the image for prediction.
        - Checks in-memory cache first (same image_id = no recompute).
        - Falls back to disk cache (restores SAM feature tensors).
        - Computes fresh embedding as last resort and saves it to disk.
        """

    @abstractmethod
    def predict_from_points(
        self,
        points: list[tuple[float, float]],  # pixel coordinates [(x, y), ...]
        labels: list[int],  # 1 = positive, 0 = negative
    ) -> tuple[object, float]:
        """
        Run mask prediction for the currently loaded image.
        Returns (binary_mask np.ndarray [H, W], confidence_score float).
        Raises RuntimeError if set_image() has not been called first.
        """

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True after load_model() has completed successfully."""
