"""Tests for the pydantic ``Settings`` object and derived paths."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import _PROJECT_ROOT, SAM_VARIANTS, Settings, settings


def test_settings_paths_are_path_objects() -> None:
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.MODELS_DIR, Path)


def test_models_dir_default_is_repo_relative() -> None:
    # Guard against re-introducing a machine-specific absolute default path:
    # the default must be derived from the repository root, not hard-coded.
    assert Settings.model_fields["MODELS_DIR"].default == _PROJECT_ROOT / "models"


def test_sam_checkpoint_path_composes_models_dir_and_filename() -> None:
    assert settings.sam_checkpoint_path == (
        settings.MODELS_DIR / settings.sam_checkpoint_name
    )


def test_sam_variant_is_configurable() -> None:
    """The documented .env knob must actually change the model that loads.

    Regression test: SAM_CHECKPOINT and SAM_CONFIG used to be module constants,
    so INSTALL.md told users to select base_plus via .env while the app silently
    kept loading tiny.
    """
    configured = Settings(SAM_VARIANT="base_plus")
    assert configured.sam_checkpoint_name == "sam2.1_hiera_base_plus.pt"
    assert configured.sam_config.endswith("sam2.1_hiera_b+.yaml")
    assert configured.sam_checkpoint_path.name == "sam2.1_hiera_base_plus.pt"


def test_every_variant_pairs_a_checkpoint_with_its_config() -> None:
    for variant, (checkpoint, config) in SAM_VARIANTS.items():
        assert checkpoint.endswith(".pt"), variant
        assert config.endswith(".yaml"), variant


def test_unknown_sam_variant_is_rejected() -> None:
    """Falling back silently would run a different model than configured."""
    with pytest.raises(ValidationError, match="not recognised"):
        Settings(SAM_VARIANT="enormous")


def test_polygon_tolerance_is_positive() -> None:
    assert settings.POLYGON_TOLERANCE > 0
