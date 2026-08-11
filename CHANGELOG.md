# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-11

### Fixed
- **Concurrent requests could return a mask from the wrong image.** The shared
  `SAM2Backend` held the selected image in mutable state while `/inference/point`
  called `set_image()` and `predict_from_points()` as two separate steps from a
  sync endpoint, which FastAPI runs in a threadpool. Added a re-entrant lock and
  a `predict_on_image()` method that performs select-and-predict atomically.
- **The out-of-VRAM handler never fired.** `torch.cuda.OutOfMemoryError`
  subclasses `RuntimeError`, not `MemoryError`, so the `except MemoryError`
  guarding the 507 response was dead code and OOM surfaced as a bare 500.
- **Cached embeddings were never deleted.** Deleting an image removed
  `_embeddings/{id}.npy` while the cache writes `{id}.pt`, so embeddings
  accumulated indefinitely. Deletion now goes through `EmbeddingCache.delete()`.
- **Upload filenames were not sanitised.** A multipart part named
  `../../../evil.png` passed the extension allow-list and was written outside
  the project directory. Filenames are now reduced to their basename.
- **`SAM_CHECKPOINT` / `SAM_CONFIG` could not be configured.** They were module
  constants, so `INSTALL.md` told users to select `base_plus` via `.env` while
  the app silently kept loading `tiny`. Replaced by a single validated
  `SAM_VARIANT` setting that selects checkpoint and config together; an unknown
  value or a missing checkpoint now fails loudly at startup.
- `start.bat` pointed at hard-coded conda environments under one developer's home
  directory. It now resolves `.venv` relative to the script and reports a usable
  error when the environment or Node is missing.
- Every dataset export leaked a full copy of the dataset into the system temp
  directory; the temporary tree is now removed once the response is streamed.
- A corrupt or oversized upload aborted the whole batch mid-write, leaving files
  on disk with no database row.

### Added
- First HTTP-level tests (`tests/test_images_api.py`), covering upload,
  path-traversal rejection, per-file skip reporting, filename de-duplication and
  embedding cleanup on delete.
- Uploads return `{"created": [...], "skipped": [{"name", "reason"}]}` instead of
  silently dropping unsupported files.
- 100 MB per-file upload cap and a structural check before decoding.

### Changed
- `INSTALL.md` rewritten in English against the current code. It previously
  referenced a dead project name (`auto_Roboflow`), the pre-rename "SAMark"
  branding, and a checkpoint path inside one developer's personal directory tree.
- `torch` is imported lazily in `EmbeddingCache`, so the images API no longer
  pulls the ML stack into a filesystem-only request path.
- API title corrected from "SAMark API" to "annotation-station API".
- The tool comparison table is now labelled a feature comparison rather than a
  benchmark, since no measurements were run.

## [0.1.0] - 2026-07-18

### Added
- Initial release: SAM 2.1-assisted, fully local annotation platform
  (FastAPI backend + React frontend).
- Three-level embedding cache (memory → disk → encode).
- Export pipeline: YOLO-seg, YOLO-det, and COCO JSON with configurable splits.
- `pyproject.toml` packaging, CI pipeline (ruff, black, mypy, pytest),
  and an initial test suite for the mask-processing and export core.
