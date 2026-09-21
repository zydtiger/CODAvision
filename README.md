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
   - [Linux with uv](#linux-with-uv)
   - [Step 1: Install Miniconda](#-step-1-install-miniconda)
   - [Step 2: Create and Activate CODAvision Environment](#-step-2-create-and-activate-codavision-environment)
   - [Step 3: Install CUDA Toolkit and cuDNN](#-step-3-install-cuda-toolkit-and-cudnn)
   - [Step 4: Install CODAvision](#-step-4-install-codavision)
   - [Step 5: Launch CODAvision GUI](#️-step-5-launch-codavision-gui)
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
  - Operating System: Windows 10/11, macOS 11+, or Linux
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

### Linux with uv

The reproducible uv environment targets Linux x86_64 with Python 3.10.
It runs TensorFlow and PyTorch in the same environment and GUI, using
TensorFlow 2.21, legacy Keras 2 (`tf-keras`), PyTorch 2.11/torchvision 0.26
with CUDA 12.8, and NumPy 1.26.4. Framework selection and model checkpoint
detection follow the existing CODAvision configuration.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
run these commands from your CODAvision checkout:

```bash
uv sync --locked --extra gpu
uv run --locked --extra gpu CODAvision
```

uv creates a local `.venv`, installs Python if needed, and selects the
PyTorch cu128 wheel index declared in `pyproject.toml`. Keep `--extra gpu`
on both commands so TensorFlow's CUDA dependencies remain installed. Avoid
mixing manual pip installations into this environment; uv synchronizes it
to `uv.lock`.

A working NVIDIA driver and desktop display are required. For Blackwell
GPUs, use an up-to-date driver supporting CUDA 12.8 (Linux driver 570.26 or
newer; see the [NVIDIA support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.8.0/reference/support-matrix.html)).
The environment supplies CUDA/cuDNN libraries as Python packages; do not
install the Conda CUDA 11.2 stack below into it. Qt may also require your
distribution's desktop libraries, including `libxcb-cursor` on X11.

TensorFlow 2.21 reports PTX JIT compilation for Blackwell, so the first use
of GPU kernels can take substantially longer than subsequent runs. This
version was selected because TensorFlow 2.20 failed basic float GPU
operations on an RTX 5090 with `CUDA_ERROR_INVALID_PTX`. TensorFlow 2.21
and PyTorch 2.11/cu128 passed small convolution forward/backward checks
in one process on that GPU with driver 595.91.07; full training and
performance have not been validated.

CODAvision enables legacy Keras and TensorFlow GPU memory growth before
its package imports. For notebooks, import `base` before importing or using
TensorFlow/Keras, or set `TF_USE_LEGACY_KERAS=1` before starting Python.
Use `from tensorflow import keras` for CODAvision models and callbacks;
the separate Keras 3 package is a TensorFlow dependency, not the API used
by CODAvision. Memory growth avoids TensorFlow reserving all GPU memory
at startup, but concurrent workloads still share the available VRAM.

Final and best TensorFlow models retain HDF5 contents under their historical
`.keras` filenames. Both save paths explicitly select HDF5: legacy Keras
previously treated the best checkpoint's `.keras` suffix as HDF5 even when
the caller requested `save_format='tf'`. Existing HDF5 checkpoints do not
need conversion.

The uv lock targets Linux only. Windows and macOS retain the Conda/pip
installation paths below.

### Conda installation

### Step 1: Install Miniconda

Download and install Miniconda by following the instructions provided [here](https://docs.anaconda.com/miniconda/).

---

### Step 2: Create and Activate CODAvision Environment

**For Windows:**
```bash
conda create -n CODAvision python=3.9
conda activate CODAvision
```

**For Linux:**
```bash
conda create -n CODAvision python=3.10
conda activate CODAvision
```

**For macOS:**

- **Apple Silicon with GPU support (M1/M2/M3/M4)** — requires Python 3.10+:
```bash
conda create -n CODAvision python=3.10
conda activate CODAvision
```
---

### Step 3: Install CUDA Toolkit and cuDNN

**For Windows only:**

Ensure that CUDA drivers are installed as per the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Then, install the CUDA Toolkit and cuDNN:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

**For macOS users:** Skip this step.

**For Linux users:** Install a compatible NVIDIA driver as described in the
uv section. The `gpu` extra below supplies CUDA/cuDNN packages; skip the
Conda CUDA toolkit installation.

---

### Step 4: Install CODAvision

> ⚠️ **Note:**  
> Ensure Git is installed. If not, download it from [here](https://git-scm.com/downloads).

**For Windows:**
```bash
pip install -e git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision
```

**For Linux:**
```bash
pip install -e "git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
```

**For macOS:**

- **Apple Silicon with GPU acceleration (M1/M2/M3/M4):**
```bash
pip install -e "git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision[macos-silicon]"
```

This installs `tensorflow-macos`, `tensorflow-metal`, and other dependencies. Do not install `keras` separately (it's included).

After installation, restart your IDE and reactivate the environment:
```bash
conda activate CODAvision
```

>💡 **Alternative installation option:**
> You can also clone the repository first and install dependencies locally:  
> ```bash
> git clone https://github.com/Kiemen-Lab/CODAvision.git
> cd CODAvision
> pip install -e .  # Windows/macOS
> ```

For a local Linux checkout in a Conda environment, use:

```bash
pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
```

> 💡 **Windows PyTorch GPU Support (Optional):**
> For the legacy Windows installation, select CUDA-enabled PyTorch *before* installing CODAvision:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -e .
> ```
---

### 🖼️ Step 5: Launch CODAvision GUI

After completing the installation, run the following command to launch the GUI:
```bash
python CODAvision.py
```

**⏱️ Typical Installation Time:** Approximately 10–15 minutes on a standard desktop computer.

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
