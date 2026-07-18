from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SAM model locked to tiny — not a pydantic field, cannot be overridden by .env
SAM_CHECKPOINT = "sam2.1_hiera_tiny.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / "backend" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage paths (override via .env). MODELS_DIR must contain the SAM checkpoint.
    DATA_DIR: Path = _PROJECT_ROOT / "data" / "projects"
    MODELS_DIR: Path = _PROJECT_ROOT / "models"

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
    def sam_checkpoint_path(self) -> Path:
        return self.MODELS_DIR / SAM_CHECKPOINT

    @field_validator("DATA_DIR", "MODELS_DIR", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)


settings = Settings()
