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
   - [Prerequisites](#prerequisites)
   - [Linux and Apple Silicon macOS](#linux-and-apple-silicon-macos)
   - [Windows](#windows)
   - [Using TiSAM](#using-tisam)
3. [Demo](#-3-demo)
   - [Sample Dataset](#sample-dataset)
   - [Instructions to Run on Sample Data](#instructions-to-run-on-sample-data)
   - [Expected Output](#expected-output)
   - [Expected Runtime](#expected-runtime)
4. [Adding Custom Model Architectures](#-4-adding-custom-model-architectures)

---

## 1. System Requirements

### 🧰 Hardware

- **Minimum Requirements:**
  - Computer with ≥16 GB RAM
  - NVIDIA GPU with ≥8 GB VRAM (Windows/Linux only)
  - Operating System: Windows 10/11, macOS (Apple Silicon, CPU development), or Linux
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

CODAvision is installed with [uv](https://docs.astral.sh/uv/), which creates the Python
environment and installs the exact locked dependencies (TensorFlow, PyTorch and TiSAM).

### Prerequisites

- [Git](https://git-scm.com/downloads) (TiSAM is installed from a pinned Git commit of `Kiemen-Lab/tisam`)
- A current [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) (Windows/Linux, for GPU use)
- **uv 0.10.10 or newer** (check with `uv --version`). On Windows with Anaconda, an older
  `uv.exe` may come first on `PATH` and fail with
  `Required uv version >=0.10.10 does not match the running version`. Run `where.exe uv`
  and call the newer copy, e.g. `Set-Alias uv "$env:CONDA_PREFIX\Scripts\uv.exe"`.

You do not need to install Python yourself: uv installs Python 3.10 (`.python-version`).

### Linux and Apple Silicon macOS

```bash
git clone https://github.com/Kiemen-Lab/CODAvision.git
cd CODAvision
uv sync --locked
uv run --locked CODAvision
```

Linux installs TensorFlow's CUDA libraries and PyTorch with CUDA 12.8. macOS runs on the
CPU (no CUDA). To use Python 3.11 instead of 3.10, run `uv python pin 3.11` before syncing.

### Windows

Native Windows uses TensorFlow 2.10, which needs the CUDA 11.2 and cuDNN 8.1 libraries
to use the GPU, and `uv` cannot install them. The commands below use conda only for this:
[environment-windows.yml](environment-windows.yml) creates a new conda environment that
contains Python 3.10, those two libraries and `uv`. The CODAvision packages themselves are
then installed by `uv`. Install [Miniconda](https://docs.anaconda.com/miniconda/) first if needed.

```bash
git clone https://github.com/Kiemen-Lab/CODAvision.git
cd CODAvision
conda env create -n codavision-gpu -f environment-windows.yml
conda activate codavision-gpu
uv sync --locked
uv run --locked CODAvision
```

Keep the conda environment activated whenever you run CODAvision, so TensorFlow can find
the CUDA libraries. PyTorch brings its own CUDA 12.8 runtime. Windows requires Python 3.10.

> **No TensorFlow GPU needed?** The PyTorch models (DeepLabV3+ PyTorch, TiSAM) do not need
> conda: on Windows you can skip the conda steps and run only `uv sync --locked` and
> `uv run --locked CODAvision`. TensorFlow models then run on the CPU.

**⏱️ Typical installation time:** 10–15 minutes.

### Which model needs what

| Model | Framework | Extra requirements |
|---|---|---|
| UNet, DeepLabV3+ (TensorFlow) | TensorFlow | Windows GPU: the conda CUDA 11.2 / cuDNN 8.1 environment above |
| DeepLabV3+ (PyTorch) | PyTorch | None beyond the install above |
| TiSAM | PyTorch | Hugging Face access to the encoder weights, SAM3 checkpoint, a small batch size (see below) |

**Choosing the framework:** the DeepLabV3+ model is trained with TensorFlow by default.
To train it with PyTorch instead, set `CODAVISION_FRAMEWORK` before launching (no config
file edit needed). TiSAM always uses PyTorch regardless of this setting.

```powershell
# Windows PowerShell
$env:CODAVISION_FRAMEWORK = "pytorch"; uv run --locked CODAvision
```

```bash
# Linux / macOS
CODAVISION_FRAMEWORK=pytorch uv run --locked CODAvision
```

### Using TiSAM

#### Prerequisite: Hugging Face access

TiSAM downloads two gated models the first time it runs: the encoder `MahmoodLab/UNI2-h`
and the SAM3 weights `facebook/sam3`. Without access, training fails when the model is
built, so complete this one-time setup before selecting TiSAM:

1. Create a [Hugging Face](https://huggingface.co/join) account.
2. Request access on the [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and
   [SAM3](https://huggingface.co/facebook/sam3) model pages and wait for approval.
3. Log in inside the CODAvision environment:
   1. Create a token: on huggingface.co open your profile, then **Settings > Access Tokens >
      Create new token**. A **Read** token is enough. Copy it when it is shown, since it
      cannot be viewed again.
   2. From the CODAvision folder, run:
      ```bash
      uv run hf auth login
      ```
   3. Paste the token when prompted (the characters are not displayed). If asked whether to
      add the token as a git credential, answer `n`.
   4. Check that it worked with `uv run hf auth whoami`, which prints your username.

   The login is stored on your computer, so you only need to do this once. If the command
   says "User is already logged in", a token is saved but may be expired or invalid: run
   `uv run hf auth whoami`, and if it reports an invalid token, run `uv run hf auth login --force`
   to replace it.

The weights are then downloaded and cached automatically. To use a SAM3 file you already
have, set `sam3_checkpoint` in the YAML settings below instead.

#### Running TiSAM

Select **TiSAM** in the model dropdown under **Advanced settings** and use the normal image,
annotation, class, whitespace and nesting tabs. Tiles, class colors, testing and
classification are shared with the other models. CODAvision's final `black` class is
treated as TiSAM's `void` class, so it is not learned as a predicted class.

- **Batch size** comes from the Advanced settings batch size box. TiSAM uses much more GPU
  memory than the CNN models, so start small and raise `accumulation_steps` to compensate.
- **Optional TiSAM settings (YAML)** appear below it when TiSAM is selected. Accepted
  sections are `model`, `training` and `data`:

  ```yaml
  model:
    extra_encoder: hf-hub:MahmoodLab/UNI2-h
  training:
    device: cuda
    epochs: 100
    learning_rate: 0.0001
    accumulation_steps: 16
    precision: bf16
    sam3_checkpoint: /path/to/sam3.pt
  ```

  Class names, colors, tile size, batch size and dataset locations come from the normal
  tabs and cannot be set here.
- **Output:** the trained model is saved as `best_model_TiSAM.safetensors` next to `net.pkl`.

### Notes

- Supported Python: 3.10 on Windows; 3.10 or 3.11 on Linux and macOS. Python 3.9 is no
  longer supported.
- Developers: `uv sync --locked --extra dev`, then `uv run --locked --extra dev pytest`.
- Without uv (`pip`) is possible with Python 3.10 (`pip install torch==2.11.0
  torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128`, then
  `pip install -e .`), but `uv.lock` is the tested path.
- In notebooks and custom models, import `base` before using Keras and use
  `from tensorflow import keras` (or set `TF_USE_LEGACY_KERAS=1`). Existing HDF5
  checkpoints keep their `.keras` filenames. Custom AdamW code on Windows must use
  `jit_compile=False`.

---


## 🎬 3. Demo

### Sample Dataset

Access the sample dataset [here](https://drive.google.com/drive/folders/1dkF10ojFylRl1OrcjRcgz0JIey1-zJwB?usp=drive_link).

### Instructions to Run on Sample Data

Access the demo instructions [here](https://drive.google.com/file/d/1ZtL0MrC_uGJmYUgUi4EBto6gyXNsg3Hh/view?usp=drive_link).

### Expected Output

Access the expected output [here](https://drive.google.com/drive/folders/1D3xujNXFZjP76CYznlfZtLYrKdyaKGDU?usp=sharing).

### Expected Runtime

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

