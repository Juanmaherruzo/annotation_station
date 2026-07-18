"""Tests for the pydantic ``Settings`` object and derived paths."""

from pathlib import Path

from app.config import _PROJECT_ROOT, SAM_CHECKPOINT, Settings, settings


def test_settings_paths_are_path_objects() -> None:
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.MODELS_DIR, Path)


def test_models_dir_default_is_repo_relative() -> None:
    # Guard against re-introducing a machine-specific absolute default path:
    # the default must be derived from the repository root, not hard-coded.
    assert Settings.model_fields["MODELS_DIR"].default == _PROJECT_ROOT / "models"


def test_sam_checkpoint_path_composes_models_dir_and_filename() -> None:
    assert settings.sam_checkpoint_path == settings.MODELS_DIR / SAM_CHECKPOINT


def test_polygon_tolerance_is_positive() -> None:
    assert settings.POLYGON_TOLERANCE > 0
