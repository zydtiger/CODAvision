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

The modern environment supports Python 3.10–3.11 on Linux x86_64 and native
Windows x86_64. `.python-version` selects Python 3.10 by default. The shared
stack uses TensorFlow/tf-keras 2.21, PyTorch 2.11, torchvision 0.26, and
NumPy 1.26.x. Python 3.12+ requires a newer PySide6 release and is not yet
included in this setup.

| Platform | TensorFlow | PyTorch wheel | Installation |
| --- | --- | --- | --- |
| Linux / Windows WSL2 | NVIDIA GPU | CUDA 12.8 | Locked cu128 environment below |
| Native Windows | CPU | CUDA 12.8 | Same locked cu128 environment |
| CPU-only Linux / Windows | CPU | CPU | Explicit `cpu` backend below |

[TensorFlow stopped native Windows CUDA support after 2.10](https://www.tensorflow.org/install/pip#windows-native).
Use WSL2 with GPU access and WSLg for the GUI when both frameworks need GPU
acceleration on Windows. The modern native Windows route replaces the old
TensorFlow 2.10 / Conda CUDA 11.2 installation; keep an existing legacy
environment separate if you still need it.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone
CODAvision, and run from the checkout (use PowerShell on Windows):

```sh
uv sync --locked --extra gpu
uv run --locked --extra gpu CODAvision
```

uv creates `.venv`, obtains Python if needed, and installs the locked
dependencies. Project index configuration selects PyTorch cu128 wheels on
both Linux and Windows. The `gpu` extra adds TensorFlow's CUDA packages on
Linux/WSL2; it does not enable TensorFlow GPU on native Windows. Keep the
extra on subsequent synchronizing `uv run` commands. To select Python 3.11,
pass `--python 3.11` to both commands.

A compatible NVIDIA driver and desktop display are required for this GPU
route. CUDA/cuDNN runtime packages come from the Python environment; do not
install the old Conda CUDA toolkit into it. On Linux, Qt may also require
distribution desktop libraries such as `libxcb-cursor` on X11. The OpenCV
dependency is headless because CODAvision displays windows through PySide6;
install only one OpenCV distribution providing the `cv2` namespace.

The Linux Python 3.10 environment passed small TensorFlow/PyTorch convolution
forward/backward checks in one process on an RTX 5090 with driver 595.91.07,
plus headless GUI and checkpoint save/load checks. Windows and Python 3.11
are included in dependency resolution; their GUI/GPU execution is not yet
validated. Full training and performance have not been validated.

TensorFlow 2.21 reports PTX JIT compilation on Blackwell, so initial GPU
kernel use can be slow. TensorFlow 2.20 failed float GPU operations on the
tested RTX 5090 with `CUDA_ERROR_INVALID_PTX`, which is why this environment
uses 2.21.

### Select a PyTorch backend

Package version requirements do not contain CUDA suffixes. For an installation
chosen for the current machine rather than the project lock, use uv's pip
interface (examples use uv 0.10.10). Start with a fresh virtual environment;
do not mix these installation routes in one environment:

```sh
uv venv --python 3.10

# Linux/WSL2: shared, tested CUDA backend for TensorFlow and PyTorch.
uv pip install --no-sources --torch-backend=cu128 -e '.[gpu]'

# Native Windows: auto-select the PyTorch backend (GPU or CPU); TensorFlow is CPU.
uv pip install --no-sources --torch-backend=auto -e .

# CPU-only machine: explicitly select CPU PyTorch wheels.
uv pip install --no-sources --torch-backend=cpu -e .

# Run the environment created by the chosen install command above.
uv run --no-sync CODAvision
```

Choose one install command for the intended environment. `--no-sources`
ignores this project's fixed PyTorch index mapping. `--torch-backend` selects
the index for PyTorch packages, including transitive dependencies; it does
not select a TensorFlow backend. `auto` inspects the local hardware/driver
and may select a different CUDA release or CPU. It does not negotiate a
shared CUDA/cuDNN stack with TensorFlow, nor switch backends to solve an
incompatible Python or package constraint. For Linux dual-framework GPU
use, select `cu128` explicitly.

These pip installations do not use or update `uv.lock`. In uv 0.10.10,
`--torch-backend` is not available on `uv sync`. A later `uv sync` or
synchronizing `uv run` can replace the selected backend; use `--no-sync`
for the manually selected environment. `UV_TORCH_BACKEND` is the equivalent
environment variable. See the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
The old `pytorch-cuda118`, `pytorch-cuda121`, `pytorch-cuda124`,
`pytorch-cuda`, and `pytorch-cpu` extras have been removed: they did not
actually select wheel indexes.

### Conda and pip

Conda remains an option for managing Python on Linux and Windows. From the
checkout, use a new environment:

```sh
conda create -n CODAvision python=3.10 pip
conda activate CODAvision
python -m pip install -e '.[gpu]' --extra-index-url https://download.pytorch.org/whl/cu128
CODAvision
```

The extra index supplies the matching CUDA 12.8 PyTorch wheels. Unlike the
uv lock, pip resolves dependencies at installation time. Native Windows
still runs TensorFlow on CPU.

### Model compatibility

CODAvision enables legacy Keras and TensorFlow GPU memory growth before its
package imports. In notebooks, import `base` before TensorFlow/Keras, or set
`TF_USE_LEGACY_KERAS=1` before starting Python. Use `from tensorflow import
keras` for models and callbacks; TensorFlow's separate Keras 3 dependency
is not CODAvision's model API. Memory growth avoids reserving all GPU memory
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
- **Desktop Computer with no GPU:** Image processing and training time may extend up to 10 hours.

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
