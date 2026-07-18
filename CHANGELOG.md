# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-18

### Added
- Initial release: SAM 2.1-assisted, fully local annotation platform
  (FastAPI backend + React frontend).
- Three-level embedding cache (memory → disk → encode).
- Export pipeline: YOLO-seg, YOLO-det, and COCO JSON with configurable splits.
- `pyproject.toml` packaging, CI pipeline (ruff, black, mypy, pytest),
  and an initial test suite for the mask-processing and export core.
