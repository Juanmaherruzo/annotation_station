# SAMark — Guía de instalación

Aplicación local de anotación de datasets con SAM 2.1 como motor de segmentación asistida.

---

## Requisitos del sistema

| Componente | Mínimo | Notas |
|---|---|---|
| OS | Windows 10/11 | El `start.bat` es Windows |
| GPU | NVIDIA con 4 GB VRAM | RTX 3050 Laptop o superior |
| Driver NVIDIA | >= 520 | [descargar](https://www.nvidia.com/drivers) |
| CUDA | 12.x | Solo el driver; PyTorch trae sus propias DLLs |
| Anaconda / Miniconda | cualquier versión reciente | [descargar](https://www.anaconda.com/download) |
| Node.js | >= 18 | Instalado vía conda (ver paso 3) |

---

## Instalación paso a paso

### 1. Clonar o copiar el proyecto

```
git clone <url-del-repo>
cd auto_Roboflow
```

---

### 2. Crear el entorno conda del backend

```bat
conda create -n sam_studio python=3.11 -y
conda activate sam_studio
```

---

### 3. Instalar Node.js en un entorno conda separado

Node.js **no debe** instalarse en `sam_studio` para evitar conflictos.
Si ya tienes un entorno con Node.js (por ejemplo `detector_copas`) omite este paso.

```bat
conda create -n detector_copas -y
conda activate detector_copas
conda install -c conda-forge nodejs -y
```

Verifica:
```bat
node --version    # >= 18.x
npm --version
```

---

### 4. Instalar SAM 2.1

```bat
conda activate sam_studio
pip install git+https://github.com/facebookresearch/sam2.git
```

Verifica que se instaló correctamente:
```bat
python -c "import sam2; print('sam2 OK')"
```

---

### 5. Instalar PyTorch con CUDA

> Requiere CUDA 12.x en el driver (comprueba con `nvidia-smi`).

```bat
conda activate sam_studio
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Si tu driver tiene CUDA 12.4 o inferior usa `cu124`:
```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verifica GPU:
```bat
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

### 6. Instalar el resto de dependencias del backend

```bat
conda activate sam_studio
cd backend
pip install -e ".[dev]"
cd ..
```

---

### 7. Instalar dependencias del frontend

```bat
conda activate detector_copas
cd frontend
npm install
cd ..
```

---

### 8. Descargar el checkpoint de SAM 2.1

Descarga **uno** de los siguientes checkpoints y colócalo en la carpeta de modelos:

| Modelo | VRAM | Velocidad | Calidad |
|---|---|---|---|
| `sam2.1_hiera_tiny.pt` | ~1.5 GB | Muy rápido | Buena |
| `sam2.1_hiera_base_plus.pt` | ~2.5 GB | Rápido | Mejor (**recomendado**) |
| `sam2.1_hiera_large.pt` | ~4 GB | Lento en 4 GB | Máxima |

Descarga desde: https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description

Coloca el `.pt` en:
```
C:\Users\<tu_usuario>\Documents\Proyectos_personales\Python\DATA_LEARNING\Modelos\
```
o edita `MODELS_DIR` en `backend/app/config.py`.

---

### 9. Configurar el backend

Edita `backend/.env` para apuntar al checkpoint que hayas descargado:

```env
# Para base_plus (recomendado con RTX 3050):
SAM_CHECKPOINT=sam2.1_hiera_base_plus.pt
SAM_CONFIG=configs/sam2.1/sam2.1_hiera_b+.yaml

# Para tiny (menos VRAM, más rápido):
# SAM_CHECKPOINT=sam2.1_hiera_tiny.pt
# SAM_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml

POLYGON_TOLERANCE=4.0
```

---

### 10. Adaptar el start.bat a tu máquina

Abre `start.bat` y comprueba que estas dos rutas son correctas:

```bat
set UVICORN=C:\Users\<tu_usuario>\anaconda3\envs\sam_studio\Scripts\uvicorn.exe
set NPM=C:\Users\<tu_usuario>\anaconda3\envs\detector_copas\npm.cmd
```

Si Anaconda está en otra ubicación (p.ej. `C:\ProgramData\anaconda3`) cámbiala aquí.

---

## Arranque

Doble clic en `start.bat`. Se abren dos ventanas de terminal y el navegador en `http://localhost:5173`.

- **Ventana Backend**: arranca FastAPI + carga SAM (~10-30 s la primera vez).
- **Ventana Frontend**: arranca Vite dev server.

---

## Estructura de entornos conda

```
anaconda3/
├── envs/
│   ├── sam_studio/        ← backend Python (FastAPI, SAM, PyTorch CUDA)
│   └── detector_copas/    ← Node.js para el frontend (npm/vite)
```

SAMark **no modifica** ningún otro entorno conda de tu máquina.

---

## Resolución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `"uvicorn" no se reconoce` | Ruta incorrecta en start.bat | Comprueba `UVICORN=` en start.bat |
| `"node" no se reconoce` | Ruta incorrecta en start.bat | Comprueba `NPM=` y el `set PATH=` en start.bat |
| `CUDA not available` | PyTorch CPU instalado | Repite el paso 5 con `--force-reinstall` |
| OOM en primera inferencia | Modelo demasiado grande | Cambia a `tiny` en `backend/.env` |
| Frontend se queda en blanco | Backend no arrancó aún | Espera 30 s y recarga; revisa la ventana Backend |
| Puerto 8000 ocupado | Otra instancia corriendo | Cierra la ventana Backend anterior |
