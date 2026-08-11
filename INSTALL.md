# annotation-station — Installation guide

A local annotation platform for image datasets, using SAM 2.1 as the
assisted-segmentation engine. Everything runs on your machine; no image ever
leaves it.

---

## System requirements

| Component | Minimum | Notes |
|---|---|---|
| OS | Windows 10/11, Linux or macOS | `start.bat` is Windows-only; on Linux/macOS start the two servers by hand (step 7) |
| GPU | NVIDIA with 4 GB VRAM | Optional but strongly recommended. Without CUDA, SAM runs on CPU and each click takes seconds instead of milliseconds |
| NVIDIA driver | >= 520 | [download](https://www.nvidia.com/drivers) |
| CUDA | 12.x | Driver only — PyTorch ships its own runtime libraries |
| Python | 3.11 or newer | |
| Node.js | >= 18 | Frontend build tooling |

---

## 1. Clone

```bash
git clone https://github.com/Juanmaherruzo/annotation_station.git
cd annotation_station
```

## 2. Create the Python environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

## 3. Install PyTorch with CUDA

Check your driver's CUDA version with `nvidia-smi`, then install the matching
build. PyTorch is not installed by the package metadata because the correct
wheel depends on your hardware.

```bash
# CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch torchvision
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. Install SAM 2.1

`sam2` is not published on PyPI, so it is installed from source:

```bash
pip install git+https://github.com/facebookresearch/sam2.git
python -c "import sam2; print('sam2 OK')"
```

## 5. Install the backend

```bash
cd backend
pip install -e ".[dev]"
cd ..
```

## 6. Install the frontend

```bash
cd frontend
npm install
cd ..
```

## 7. Download a SAM 2.1 checkpoint

Download **one** checkpoint from the
[SAM 2.1 model list](https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description)
and place it in the `models/` directory at the repository root:

| Variant | Checkpoint file | VRAM | Notes |
|---|---|---|---|
| `tiny` | `sam2.1_hiera_tiny.pt` | ~1.5 GB | Default. Fast, good quality |
| `small` | `sam2.1_hiera_small.pt` | ~2 GB | |
| `base_plus` | `sam2.1_hiera_base_plus.pt` | ~2.5 GB | Better masks, recommended if you have the VRAM |
| `large` | `sam2.1_hiera_large.pt` | ~4 GB | Best quality, slow on a 4 GB card |

```
annotation_station/
└── models/
    └── sam2.1_hiera_tiny.pt
```

## 8. Configure the backend

Copy the example environment file and edit it:

```bash
cp backend/.env.example backend/.env
```

The only value most people need to change is the model size, which must match
the checkpoint you downloaded in step 7:

```env
SAM_VARIANT=tiny        # tiny | small | base_plus | large
```

`SAM_VARIANT` selects the checkpoint **and** its Hydra config together, so the
two cannot fall out of step. If the matching `.pt` file is not in `models/`, the
server refuses to start and names the file it expected — it will not quietly
fall back to a different model.

## 9. Start

**Windows.** Double-click `start.bat`, or run it from a terminal. It opens two
console windows (backend and frontend) and your browser at
`http://localhost:5173`. If your virtual environment is not at `.venv`, edit the
two paths at the top of the script.

**Linux / macOS**, or to run the servers manually:

```bash
# terminal 1
cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal 2
cd frontend && npm run dev
```

The backend takes 10–30 seconds on first start while SAM loads.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SAM checkpoint not found` at startup | `SAM_VARIANT` does not match the file in `models/` | The error names the expected filename — download it or change `SAM_VARIANT` |
| `SAM_VARIANT=... is not recognised` | Typo in `backend/.env` | Use one of `tiny`, `small`, `base_plus`, `large` |
| `torch.cuda.is_available()` is `False` | CPU-only PyTorch installed | Redo step 3 with the right index URL and `--force-reinstall` |
| HTTP 507 on first click | Out of GPU memory | Switch to a smaller `SAM_VARIANT`, or close other GPU applications |
| Frontend loads but every request fails | Backend not up yet | Wait for the backend window to print that SAM is ready, then reload |
| Port 8000 already in use | A previous instance is still running | Close it, or set `PORT` in `backend/.env` |
| Masks look coarse | Polygon simplification too aggressive | Lower `POLYGON_TOLERANCE` in `backend/.env` |

---

## What gets stored, and where

Everything stays inside the repository directory:

```
data/projects/<project_id>/
├── images/       # your uploaded images
├── thumbnails/   # generated previews
└── _embeddings/  # cached SAM feature tensors (safe to delete; they are recomputed)
```

`data/` and `models/` are excluded from version control. No telemetry is
collected and the application makes no outbound network requests.
