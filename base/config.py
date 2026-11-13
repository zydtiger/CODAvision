"""
Centralized Configuration Module for CODAvision

This module provides centralized configuration management for the CODAvision
package, eliminating magic numbers and hardcoded values throughout the codebase.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class ThresholdDefaults:
    """Default values for tissue area threshold selection."""
    HE_DEFAULT = 205  # Default threshold for H&E stained images
    GRAYSCALE_DEFAULT = 128  # Default threshold for grayscale images
    BATCH_SIZE = 10  # Number of images to process in each batch
    BATCH_DELAY_MS = 100  # Delay between batches in milliseconds


class DisplayDefaults:
    """Default values for display and visualization."""
    MAX_WIDTH = 1500  # Maximum display width in pixels
    MAX_HEIGHT = 780  # Maximum display height in pixels
    REGION_SIZE = 600  # Default region size for threshold selection
    SAMPLE_SIZE = 20  # Default number of images to sample


class FileExtensions:
    """Supported file extensions for different image types."""
    TIFF_EXTENSIONS = ['.tif', '.tiff']
    STANDARD_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    ALL_IMAGE_EXTENSIONS = TIFF_EXTENSIONS + STANDARD_EXTENSIONS
    
    @classmethod
    def get_search_patterns(cls) -> List[str]:
        """Get glob patterns for all supported image types."""
        return ['*' + ext for ext in cls.ALL_IMAGE_EXTENSIONS]


class ModelDefaults:
    """Default values for model training and architecture."""
    INPUT_SIZE = 512  # Default input size for segmentation models
    NUM_FILTERS = 256  # Default number of filters in convolution blocks
    KERNEL_SIZE = 3  # Default kernel size for convolutions

    # Training defaults
    BATCH_SIZE = 8  # Default batch size for training
    LEARNING_RATE = 5e-4  # Default initial learning rate
    EPOCHS = 100  # Default number of training epochs

    # Optimizer defaults
    OPTIMIZER_EPSILON = 1e-8  # Epsilon value for Adam and AdamW optimizers (for numerical stability)

    # Early stopping and learning rate reduction
    ES_PATIENCE = 6  # Patience for early stopping (epochs without improvement)
    LR_PATIENCE = 1  # Patience for learning rate reduction
    LR_FACTOR = 0.75  # Factor for learning rate reduction
    MIN_LR = 1e-7  # Minimum learning rate

    # Validation
    VALIDATION_FREQUENCY = 128  # Number of iterations between validations

    # Tile Generation
    TILE_GENERATION_MODE = "modern"  # Default tile generation mode ("modern" or "legacy")

    # ===== FRAMEWORK CONFIGURATION =====
    # "pytorch" or "tensorflow"; the CODAVISION_FRAMEWORK environment variable overrides it
    DEFAULT_FRAMEWORK = os.environ.get("CODAVISION_FRAMEWORK") or "tensorflow"

    # PyTorch-specific settings
    PYTORCH_DEVICE = "auto"  # auto, cuda, mps, cpu
    PYTORCH_COMPILE = False  # torch.compile() optimization (PyTorch 2.0+)
    PYTORCH_AMP = False  # Automatic Mixed Precision
    GRADIENT_ACCUMULATION_STEPS = 1  # Gradient accumulation for larger effective batch size

    # Model availability matrix
    TENSORFLOW_MODELS = ["DeepLabV3_plus", "UNet"]
    PYTORCH_MODELS = ["DeepLabV3_plus", "TiSAM"]


class TiSAMDefaults:
    """
    Training defaults for the TiSAM architecture.

    TiSAM is a query-decoder architecture with frozen foundation-model
    encoders, and its upstream defaults differ substantially from the
    CNN-oriented values in ModelDefaults. These are applied by
    PyTorchTiSAMTrainer, while the training loop itself stays CODAvision's.

    Source: https://github.com/Kiemen-Lab/tisam (src/tisam/config/run.py).
    """

    LEARNING_RATE = 1e-4          # vs 5e-4 for the CNN models
    WEIGHT_DECAY = 1e-2           # AdamW-style decoupled decay
    ACCUMULATION_STEPS = 5        # optimizer step every 5 batches
    WARMUP_RATIO = 0.05           # linear warmup over the first 5% of steps

    # Tiles per forward pass. The effective batch is this value times
    # ACCUMULATION_STEPS, so 3 x 5 = 15 tiles per optimizer step, which keeps
    # the optimizer seeing roughly what it saw at 1 x 16 while the GPU gets
    # three tiles at a time instead of one. A ViT this size leaves most of the
    # card idle at batch 1; raise this until VRAM is the limit, and move
    # ACCUMULATION_STEPS the other way to hold the effective batch steady.
    BATCH_SIZE = 3

    PRECISION = "bf16"

    # Upstream uses 0 workers, but its pipeline reads whole slides through
    # zarr while CODAvision reads individual tiles from disk. Measured here, a
    # GPU step takes 0.21s against 2.6s per observed iteration, so ~92% of the
    # time is spent loading data in the main process. Parallel workers remove
    # that.
    NUM_WORKERS = 14

    # Early stopping. The patience is in epochs, and the training loop turns
    # it into a number of validations (patience x validations_per_epoch).
    #
    # MIN_DELTA is the size of improvement that counts. Without it the test is
    # a bare 'val_loss < best_val_loss', so once the model converges and the
    # scheduler has cut the LR to its floor, a drift of 0.0001 in the fourth
    # decimal still resets the counter and training never stops. Observed on
    # this dataset: 110 validations required, longest stall reached 24, while
    # the loss moved 0.0008 in two hours. 0.001 is the same tolerance TiSAM's
    # own EarlyStopping uses, and it is comparable to the relative threshold
    # already given to ReduceLROnPlateau.
    #
    # The gate is cumulative, not per-validation: best_val_loss only moves
    # when the tolerance is cleared, so steady improvements of less than
    # MIN_DELTA each still count once they add up past it.
    EARLY_STOPPING_PATIENCE = 10      # epochs
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Learning-rate plateau handling. The shared default reduces after a
    # single non-improving validation, so two in a row are enough to cut the
    # LR; against a noisy loss that fires almost continuously, and at factor
    # 0.75 it takes only 25 reductions to fall from 1e-4 to a 1e-7 floor,
    # where the model stops moving at all while still costing full price per
    # epoch. A longer patience and a higher floor keep the run alive until
    # early stopping is the thing that ends it.
    LR_REDUCTION_PATIENCE = 3     # validations
    MIN_LR = 1e-6

    # Epochs between refreshes of the rolling '<model_type>.pth' checkpoint.
    # A TiSAM state_dict carries the frozen encoders too (~6 GB: 2.73 GB of
    # UNI2-h plus 3.45 GB of SAM3), and the default cadence rewrites it at
    # every validation, 11 times an epoch on the observed setup. A new best is
    # still written the moment it appears, so nothing is lost by spacing this
    # out; it only affects how current the resume point is.
    CHECKPOINT_EVERY_EPOCHS = 5

    # Loss reduction. 'per_class_mean' averages within each class before
    # applying the median-frequency weights, which balances the classes twice:
    # the per-class average already removes frequency, and the weight removes
    # it again. On this dataset that collapsed training to 2 of 13 classes.
    # 'weighted_sum' normalizes by the summed pixel weights instead, balancing
    # exactly once, and is what reached 0.8856 validation accuracy.
    LOSS_REDUCTION = "weighted_sum"
    WEIGHT_POWER = 1.0            # 1.0 = plain median-frequency weighting


class LoggingConfig:
    """Configuration for logging system."""
    LOG_ROTATION_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_ROTATION_BACKUP_COUNT = 5  # Keep 5 backup files
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEBUG_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


@dataclass
class GPUConfig:
    """Configuration for GPU usage and memory management."""
    memory_growth: bool = True  # Allow GPU memory growth
    memory_limit: float = 0.9  # Use up to 90% of GPU memory
    multi_gpu_strategy: str = 'mirrored'  # Strategy for multi-GPU training
    mixed_precision: bool = False  # Use mixed precision training


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    prefetch_buffer_size: int = 2  # Prefetch buffer size for tf.data
    shuffle_buffer_size: int = 1000  # Shuffle buffer size
    num_parallel_calls: int = -1  # Use tf.data.AUTOTUNE
    cache: bool = True  # Cache dataset in memory if possible


@dataclass
class RuntimeConfig:
    """Runtime configuration that can be modified at runtime."""
    debug_mode: bool = False
    verbose: bool = True
    use_gpu: bool = True
    num_workers: int = 4  # Number of parallel workers
    seed: int = 42  # Random seed for reproducibility
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass
class TileGenerationConfig:
    """Configuration for tile generation behavior."""
    mode: str = "modern"  # "modern" or "legacy" or custom name
    reduction_factor: int = 10
    use_disk_filter: bool = False
    crop_rotations: bool = False
    class_rotation_frequency: int = 5
    deterministic_seed: Optional[int] = 3
    big_tile_size: int = 10240
    file_format: str = "png"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate mode (allow custom modes for flexibility)
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError(f"mode must be a non-empty string, got '{self.mode}'")

        # Validate reduction_factor
        if not 1 <= self.reduction_factor <= 20:
            raise ValueError(f"reduction_factor must be 1-20, got {self.reduction_factor}")

        # Validate file_format and normalize to 3-char extension
        if not isinstance(self.file_format, str):
            raise ValueError(f"file_format must be a string, got {type(self.file_format)}")

        # Normalize file format to 3-char extension
        format_lower = self.file_format.lower()
        if format_lower in ("tif", "tiff"):
            self.file_format = "tif"
        elif format_lower == "png":
            self.file_format = "png"
        elif format_lower in ("jpg", "jpeg"):
            self.file_format = "jpg"
        else:
            raise ValueError(f"file_format must be 'tif'/'tiff', 'png', or 'jpg'/'jpeg', got '{self.file_format}'")


# Preset tile generation configurations
MODERN_CONFIG = TileGenerationConfig(
    mode="modern",
    reduction_factor=10,
    use_disk_filter=False,
    crop_rotations=False,
    class_rotation_frequency=5,
    deterministic_seed=3,
    big_tile_size=10240,
    file_format="png"
)

LEGACY_CONFIG = TileGenerationConfig(
    mode="legacy",
    reduction_factor=5,
    use_disk_filter=True,
    crop_rotations=True,
    class_rotation_frequency=3,
    deterministic_seed=None,
    big_tile_size=10000,
    file_format="tif"
)


# Global runtime configuration instance
runtime_config = RuntimeConfig()


def get_threshold_config(mode: str = 'HE') -> int:
    """
    Get the default threshold value for a given mode.
    
    Args:
        mode: The threshold mode ('HE' or 'grayscale')
        
    Returns:
        Default threshold value for the specified mode
    """
    if mode.upper() == 'HE':
        return ThresholdDefaults.HE_DEFAULT
    else:
        return ThresholdDefaults.GRAYSCALE_DEFAULT


def get_display_config() -> dict:
    """
    Get display configuration as a dictionary.
    
    Returns:
        Dictionary containing display configuration values
    """
    return {
        'max_width': DisplayDefaults.MAX_WIDTH,
        'max_height': DisplayDefaults.MAX_HEIGHT,
        'region_size': DisplayDefaults.REGION_SIZE,
        'sample_size': DisplayDefaults.SAMPLE_SIZE
    }


def get_model_config() -> dict:
    """
    Get model configuration as a dictionary.

    Returns:
        Dictionary containing model configuration values
    """
    return {
        'input_size': ModelDefaults.INPUT_SIZE,
        'num_filters': ModelDefaults.NUM_FILTERS,
        'kernel_size': ModelDefaults.KERNEL_SIZE,
        'batch_size': ModelDefaults.BATCH_SIZE,
        'learning_rate': ModelDefaults.LEARNING_RATE,
        'epochs': ModelDefaults.EPOCHS,
        'es_patience': ModelDefaults.ES_PATIENCE,
        'lr_patience': ModelDefaults.LR_PATIENCE,
        'lr_factor': ModelDefaults.LR_FACTOR,
        'min_lr': ModelDefaults.MIN_LR,
        'validation_frequency': ModelDefaults.VALIDATION_FREQUENCY
    }


def get_default_tile_config() -> TileGenerationConfig:
    """
    Get default tile configuration from ModelDefaults.

    Returns:
        TileGenerationConfig: Either MODERN_CONFIG or LEGACY_CONFIG preset

    Raises:
        ValueError: If the mode is not 'modern' or 'legacy'
    """
    mode = ModelDefaults.TILE_GENERATION_MODE.lower()

    if mode == "legacy":
        return LEGACY_CONFIG
    elif mode == "modern":
        return MODERN_CONFIG
    else:
        raise ValueError(f"Invalid TILE_GENERATION_MODE: '{mode}'. Must be 'modern' or 'legacy'.")


def get_framework_config() -> dict:
    """
    Get framework configuration from ModelDefaults.

    Returns:
        Dictionary with framework configuration

    Raises:
        ValueError: If framework is not 'tensorflow' or 'pytorch'
    """
    framework = ModelDefaults.DEFAULT_FRAMEWORK.lower()

    if framework not in ['tensorflow', 'pytorch']:
        raise ValueError(f"Invalid framework: {framework}. Must be 'tensorflow' or 'pytorch'")

    return {
        'framework': framework,
        'pytorch_device': ModelDefaults.PYTORCH_DEVICE,
        'pytorch_compile': ModelDefaults.PYTORCH_COMPILE,
        'pytorch_amp': ModelDefaults.PYTORCH_AMP,
        'gradient_accumulation_steps': ModelDefaults.GRADIENT_ACCUMULATION_STEPS,
    }

class FrameworkConfig:
    """
    Framework configuration manager for PyTorch/TensorFlow switching.

    Provides a unified API for getting and setting the deep learning framework.
    """

    @staticmethod
    def get_framework() -> str:
        """
        Get the current framework ('pytorch' or 'tensorflow').

        Returns:
            Current framework name

        Example:
            >>> framework = FrameworkConfig.get_framework()
            >>> print(framework)
            'pytorch'
        """
        config = get_framework_config()
        return config['framework']

    @staticmethod
    def get_device() -> str:
        """Get the PyTorch device setting."""
        config = get_framework_config()
        return config['pytorch_device']

    @staticmethod
    def is_pytorch_compile_enabled() -> bool:
        """Check if PyTorch compilation is enabled."""
        config = get_framework_config()
        return config['pytorch_compile']

    @staticmethod
    def is_amp_enabled() -> bool:
        """Check if Automatic Mixed Precision is enabled."""
        config = get_framework_config()
        return config['pytorch_amp']

    @staticmethod
    def get_gradient_accumulation_steps() -> int:
        """Get gradient accumulation steps."""
        config = get_framework_config()
        return config['gradient_accumulation_steps']

    @staticmethod
    def set_framework(framework: str) -> None:
        """
        Set the active deep learning framework.

        Args:
            framework: Framework name ('pytorch' or 'tensorflow')

        Raises:
            ValueError: If framework is not 'tensorflow' or 'pytorch'
        """
        framework = framework.lower()
        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError(f"Invalid framework: {framework}. Must be 'tensorflow' or 'pytorch'")
        ModelDefaults.DEFAULT_FRAMEWORK = framework

    @staticmethod
    def get_all_config() -> dict:
        """Get complete framework configuration."""
        return get_framework_config()
