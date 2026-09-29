# CODAvision
[![Nature Protocols](https://img.shields.io/badge/Nature%20Protocols-s41596--026--01404--3-purple)](https://www.nature.com/articles/s41596-026-01404-3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

CODAvision is an open-source Python package designed for semantic segmentation of biomedical images through a user-friendly interface.

---

## Table of Contents

1. [System Requirements](#-1-system-requirements)
   - [Hardware](#️-hardware)
   - [Software](#-software)
2. [Installation Guide](#️-2-installation-guide)
   - [Install with uv](#install-with-uv)
   - [Native Windows GPU](#native-windows-gpu)
   - [Select a PyTorch backend](#select-a-pytorch-backend)
   - [Conda and pip](#conda-and-pip)
   - [Model compatibility](#model-compatibility)
3. [Demo](#-3-demo)
   - [Sample Dataset](#-sample-dataset)
   - [Instructions to Run on Sample Data](#-instructions-to-run-on-sample-data)
   - [Expected Output](#-expected-output)
   - [Expected Runtime](#-expected-runtime)
4. [Adding Custom Model Architectures](#-4-adding-custom-model-architectures)

---

## 1. System Requirements

### 🧰 Hardware

- **Minimum Requirements:**
  - Computer with ≥16 GB RAM
  - NVIDIA GPU with ≥8 GB VRAM (Windows/Linux only)
  - Operating System: Windows 10/11 or Linux
  - Storage: ≥2.5 GB free space

- **Tested Configuration:**
  - Workstation with 128 GB RAM
  - NVIDIA GeForce RTX 4090 GPU
  - Operating System: Windows 11

### 🖥️ Software

- [CODAvision Repository](https://github.com/Kiemen-Lab/CODAvision)
- Python IDE (optional, e.g., PyCharm, Visual Studio, Spyder)
- Image Annotation Tool (choose one):
  - [Aperio ImageScope](https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope)
  - [QuPath](https://qupath.github.io)
    
    > ⚠️ **Note for QuPath Users:**  
    > To use the GUI-guided workflow in CODAvision with annotations created in QuPath, you must first export the annotations for each image as GeoJSON files via `File > Export Objects as GeoJSON`.  
    > These GeoJSON files must then be converted into XML format, which is compatible with CODAvision.  
    > You can perform this conversion using the scripts provided in the following repository: [GeoJSON2XML](https://github.com/Kiemen-Lab/GeoJSON2XML).

---

## ⚙️ 2. Installation Guide

### Install with uv

NVIDIA GPU execution is the supported model workflow on Linux x86_64 and
native Windows x86_64. `.python-version` selects Python 3.10 by default.
Install the `gpu` extra to get both model frameworks: PyTorch/torchvision
and the platform-specific TensorFlow/Keras stack. Base dependencies contain
the shared GUI, image-processing, and scientific packages; installing the
base package alone does not provide a runnable CODAvision application.

PyTorch uses CUDA 12.8 wheels on both platforms; TensorFlow has separate
platform requirements:

| Platform | Python | TensorFlow / Keras | TensorFlow GPU runtime | PyTorch / torchvision |
| --- | --- | --- | --- | --- |
| Linux | 3.10–3.11 | TensorFlow 2.21 / tf-keras 2.21 | Python CUDA 12 packages via the `gpu` extra | 2.11 / 0.26, cu128 |
| Native Windows | 3.10 | TensorFlow 2.10.1 / Keras 2.10 | CUDA 11.2 / cuDNN 8.1, installed separately | 2.11 / 0.26, cu128 |

Both platforms use NumPy 1.26.x. Windows Python 3.11 is excluded because
TensorFlow 2.10 has no wheel for it. Python 3.12+ also requires a newer
PySide6 release and is not included in this setup.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone
CODAvision. On Linux, run from the checkout:

```sh
uv sync --locked --extra gpu
uv run --locked --extra gpu CODAvision
```

uv creates `.venv`, obtains Python if needed, and installs the locked
dependencies. The `gpu` extra selects both frameworks, platform markers
select the TensorFlow/Keras versions, and the project index configuration
selects PyTorch cu128 wheels. On Linux, the extra also installs TensorFlow's
CUDA packages; native Windows still needs its external legacy runtime below.
Keep `--extra gpu` on subsequent synchronizing `uv run` commands. To select
Python 3.11 on Linux, pass `--python 3.11` to both commands.

The `dev` and `test` extras add tools to the GPU runtime. For example:

```sh
uv sync --locked --extra gpu --extra dev
uv run --locked --extra gpu --extra dev pytest
```

Use `--extra test` instead of `--extra dev` for the minimal test tools.

A compatible NVIDIA driver and desktop display are required. On Linux, the
Python environment supplies the CUDA/cuDNN runtime; do not install the
Windows legacy CUDA toolkit into it. Qt may also require distribution
desktop libraries such as `libxcb-cursor` on X11. OpenCV is headless because
CODAvision displays windows through PySide6; install only one OpenCV
package providing the `cv2` namespace.

The Linux Python 3.10 environment passed small TensorFlow/PyTorch convolution
forward/backward checks in one process on an RTX 5090 with driver 595.91.07,
plus headless GUI and checkpoint save/load checks. TensorFlow 2.21 reports
PTX JIT compilation on Blackwell, so initial kernel use can be slow.
TensorFlow 2.20 failed float GPU operations on that RTX 5090 with
`CUDA_ERROR_INVALID_PTX`, which is why Linux uses 2.21. Python 3.11 is
included in dependency resolution; its GUI/GPU execution, full training,
and performance have not been validated.

### Native Windows GPU

Windows runs CODAvision directly, without WSL2. It retains the legacy
TensorFlow GPU runtime because [TensorFlow 2.10 was the last release with
native Windows CUDA support](https://www.tensorflow.org/install/pip#windows-native).
The [official Windows build matrix](https://www.tensorflow.org/install/source_windows#gpu)
pairs it with CUDA 11.2 and cuDNN 8.1. Modern PyTorch supplies its own CUDA
12.8 libraries; its wheel does not supply TensorFlow's legacy runtime.

For the simplest setup, use Conda to provide Python and the legacy NVIDIA
libraries, and uv to install Python packages. In an initialized Conda
PowerShell, from the checkout:

```powershell
conda create -n CODAvision -c conda-forge python=3.10 pip cudatoolkit=11.2 cudnn=8.1.0
conda activate CODAvision
uv pip install --python "$env:CONDA_PREFIX\python.exe" --no-sources --torch-backend=cu128 -e ".[gpu]"
CODAvision
```

This installs into the Conda environment and does not use `uv.lock`.
Activation makes Conda's `Library\bin` available to TensorFlow. Keep the
current NVIDIA display driver; these runtime libraries do not replace it.
Do not copy CUDA or cuDNN DLLs into `System32` or replace the DLLs bundled
with PyTorch.

To use uv without Conda, first install the CUDA 11.2 toolkit and cuDNN 8.1
from the NVIDIA archives linked in the TensorFlow guide. Make their `bin`
directories available in the launching PowerShell session, then use the
locked environment. Adjust the example cuDNN path to its extraction location:

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;C:\tools\cudnn-8.1\bin;$env:PATH"
uv sync --python 3.10 --locked --extra gpu
uv run --python 3.10 --locked --extra gpu CODAvision
```

The Windows Python 3.10 dependency set resolves with TensorFlow 2.10.1,
Keras 2.10, NumPy 1.26.4, protobuf 3.19.6, and PyTorch 2.11/cu128. Native
Windows GUI/GPU execution and coexistence of both CUDA library families
still require verification on the target RTX 20–40 series machine. In
particular, RTX 40-series operation relies on compatibility with older
CUDA kernels; the [NVIDIA Ada compatibility guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html)
describes the binary/PTX requirements. This legacy TensorFlow profile does
not establish Blackwell support. If TensorFlow reports no GPU, check the
legacy runtime installation; CPU fallback does not satisfy this setup.

### Select a PyTorch backend

Package version requirements do not contain CUDA suffixes. Outside the lock,
use an explicit backend when installing the two frameworks together:

```sh
uv venv --python 3.10
uv pip install --no-sources --torch-backend=cu128 -e '.[gpu]'
uv run --no-sync CODAvision
```

On Windows, install the legacy TensorFlow runtime first as described above.
`--no-sources` ignores the project's fixed index mapping and lets
`--torch-backend=cu128` select the PyTorch wheels. These pip installations
neither use nor update `uv.lock`; use `--no-sync` when running them so a
project sync does not replace their resolved dependencies.

In uv 0.10.10, `--torch-backend` is available only through `uv pip`.
`--torch-backend=auto` selects a PyTorch index from the local hardware/driver
and can fall back to CPU. It does not coordinate TensorFlow's runtime, so
it is not the installation route for this GPU-required setup. See the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

### Conda and pip

Conda remains an option for managing Python. Native Windows requires the
legacy runtime creation command above. On Linux, create a fresh environment:

```sh
conda create -n CODAvision python=3.10 pip
conda activate CODAvision
```

In either activated environment, ordinary pip can replace the uv installer:

```sh
python -m pip install -e '.[gpu]' --extra-index-url https://download.pytorch.org/whl/cu128
CODAvision
```

The extra index supplies CUDA 12.8 PyTorch wheels. Unlike the uv lock, pip
resolves dependencies at installation time. Neither pip nor uv installs
Windows CUDA 11.2/cuDNN 8.1 through the `gpu` extra; those libraries must
already be present from the native Windows setup above.

### Model compatibility

CODAvision enables legacy Keras and TensorFlow GPU memory growth before its
package imports. TensorFlow 2.10 already uses Keras 2; Linux TensorFlow 2.21
uses the matching `tf-keras` package. In notebooks, import `base` before
TensorFlow/Keras, or set `TF_USE_LEGACY_KERAS=1` before starting Python.
Use `from tensorflow import keras` for models and callbacks. On Linux,
TensorFlow also installs Keras 3 as a dependency; CODAvision uses the legacy
Keras API instead. Memory growth avoids reserving all GPU memory
at startup; concurrent workloads still share available VRAM.

Best and final TensorFlow checkpoints keep their historical `.keras` names
and HDF5 contents. Both save paths explicitly select HDF5; legacy Keras
previously interpreted the best checkpoint's suffix as HDF5 even when the
caller requested `save_format='tf'`. Existing HDF5 checkpoints need no
conversion.

---

## 🎬 3. Demo

### 📂 Sample Dataset

Access the sample dataset [here](https://drive.google.com/drive/folders/1dkF10ojFylRl1OrcjRcgz0JIey1-zJwB?usp=drive_link).

### 📝 Instructions to Run on Sample Data

Access the demo instructions [here](https://drive.google.com/file/d/1ZtL0MrC_uGJmYUgUi4EBto6gyXNsg3Hh/view?usp=drive_link).

### 📊 Expected Output

Access the expected output [here](https://drive.google.com/drive/folders/1D3xujNXFZjP76CYznlfZtLYrKdyaKGDU?usp=sharing).

### ⏳ Expected Runtime

- **GPU-Powered Workstation:** Approximately 2–3 hours for model training and image processing.

---

## 🔧 4. Adding Custom Model Architectures

### Adding Custom Model Architectures

CODAvision uses a flexible plugin-based architecture that allows you to easily integrate new segmentation models.

To add your own model architecture:

1. Review the comprehensive guide in [MODEL_PLUGIN_ARCHITECTURE.md](docs/MODEL_PLUGIN_ARCHITECTURE.md)
2. Follow the abstract base class pattern to ensure compatibility
3. Register your model in the factory function
4. Your model will automatically appear in the GUI and training pipeline

The plugin architecture supports:
- TensorFlow/Keras models (DeepLabV3+, UNet)
- PyTorch models (DeepLabV3+) with Keras-compatible adapter
- Multi-framework model registry
- Seamless integration with existing workflows

---

For comprehensive guidance on annotation dataset creation, see the [CODAvision Protocol](https://www.nature.com/articles/s41596-026-01404-3).

---
