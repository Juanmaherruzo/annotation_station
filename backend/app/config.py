from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# A checkpoint and its Hydra config must match, so they are chosen together by a
# single variant name rather than set as two independent strings that can drift.
SAM_VARIANTS: dict[str, tuple[str, str]] = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / "backend" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage paths (override via .env). MODELS_DIR must contain the SAM checkpoint.
    DATA_DIR: Path = _PROJECT_ROOT / "data" / "projects"
    MODELS_DIR: Path = _PROJECT_ROOT / "models"

    # SAM 2.1 size. "tiny" runs comfortably in 4 GB of VRAM; "base_plus" is more
    # accurate and needs roughly 2.5 GB more. Whatever is set here must have its
    # checkpoint present in MODELS_DIR.
    SAM_VARIANT: str = "tiny"

    # Empty string triggers auto-detection: f"cuda:{device_count()-1}" at engine init
    CUDA_DEVICE: str = ""

    # API server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    # Image processing
    THUMBNAIL_SIZE: tuple[int, int] = (256, 256)

    # Polygon simplification tolerance (pixels)
    POLYGON_TOLERANCE: float = 1.5

    @property
    def sam_checkpoint_name(self) -> str:
        """Checkpoint filename for the selected variant."""
        return SAM_VARIANTS[self.SAM_VARIANT][0]

    @property
    def sam_config(self) -> str:
        """Hydra config path for the selected variant."""
        return SAM_VARIANTS[self.SAM_VARIANT][1]

    @property
    def sam_checkpoint_path(self) -> Path:
        return self.MODELS_DIR / self.sam_checkpoint_name

    @field_validator("DATA_DIR", "MODELS_DIR", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)

    @field_validator("SAM_VARIANT")
    @classmethod
    def _known_variant(cls, v: str) -> str:
        """Reject an unknown variant loudly instead of ignoring it.

        Silently falling back would mean the app runs a different model from the
        one the user configured, and reports accuracy accordingly.
        """
        if v not in SAM_VARIANTS:
            raise ValueError(
                f"SAM_VARIANT={v!r} is not recognised. "
                f"Choose one of: {', '.join(sorted(SAM_VARIANTS))}."
            )
        return v


settings = Settings()
