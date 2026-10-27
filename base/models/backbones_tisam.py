"""TiSAM plugin for CODAvision's annotation, training and classification pipeline."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from base.models.base import BaseSegmentationModelInterface, Framework


def parse_options(text):
    """Read optional native settings; GUI-owned geometry/classes stay authoritative."""
    try:
        options = yaml.safe_load(text) if text.strip() else {}
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid TiSAM YAML: {error}") from error
    if not isinstance(options, dict) or set(options) - {"model", "training", "data"}:
        raise ValueError("TiSAM settings accept model, training and data mappings")
    reserved = {
        "model": {"num_classes", "input_hw", "output_hw"},
        "training": {"model", "data", "out_dir", "name", "batch_size"},
        "data": {"class_names", "colormap", "train", "validation", "test"},
    }
    for section, blocked in reserved.items():
        values = options.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        if set(values) & blocked:
            raise ValueError(f"{section}: {', '.join(sorted(set(values) & blocked))} come from the normal tabs")
    # Validate native field names and values without constructing encoders.
    native_config({"sxy": 512, "classNames": ["tissue", "black"], "batch_size": 1}, options)
    return options


def options_from_metadata(metadata):
    options = parse_options(metadata.get("tisam_options_yaml", ""))
    options["class_names"] = list(metadata["classNames"])
    return options


def native_config(metadata, options, model_path=None):
    """Reuse generated tiles, translating only black/void label identity."""
    from tisam.config.run import TrainConfig

    names = list(metadata["classNames"])
    if len(names) < 2 or names[-1].lower() != "black" or "void" in names[:-1]:
        raise ValueError("TiSAM requires semantic classes followed by black; void is reserved")
    canonical = ["void", *names[:-1]]
    size = int(metadata["sxy"])
    payload = deepcopy(options.get("training", {}))
    payload.setdefault("precision", "fp32")
    payload["batch_size"] = int(metadata["batch_size"])
    payload["model"] = {
        "extra_encoder": "hf-hub:MahmoodLab/UNI2-h",
        **options.get("model", {}),
        "num_classes": len(names) - 1,
        "input_hw": [size, size], "output_hw": [size, size],
    }
    payload["data"] = {**options.get("data", {}), "class_names": canonical}
    if model_path is not None:
        root = Path(model_path).resolve()
        payload.update(out_dir=str(root), name="tisam-training")
        # Tile generation converts annotations to zero-based classes; black is last.
        labels = [*names[:-1], "void"]
        for split, folder in (("train", "training"), ("validation", "validation")):
            payload["data"][split] = [{
                "path": str(root / folder),
                "metadata": {
                    "name": "codavision", "label_to_class_name": labels,
                    "supervision_ignored_raw_labels": [],
                    "metric_ignored_raw_labels": [len(names) - 1],
                    "train_void_canvas_policy": "preserve",
                },
            }]
    return TrainConfig.model_validate(payload)


class TiSAMSegmentationModel(BaseSegmentationModelInterface):
    """Convert NHWC RGB and native void-first logits to CODAvision's class order."""

    def __init__(self, input_size, num_classes, l2_regularization_weight=0, *, options=None):
        super().__init__(input_size, num_classes, l2_regularization_weight)
        self.options = deepcopy(options or {})
        names = self.options.get("class_names", [f"class_{i}" for i in range(num_classes - 1)] + ["black"])
        self.config = native_config(
            {"sxy": input_size, "classNames": names, "batch_size": 1}, self.options,
        )
        self.model = None

    def _get_framework(self):
        return Framework.PYTORCH

    def build_model(self):
        if self.model is None:
            from tisam import TiSAM
            self.model = TiSAM(self.config.model, sam3_checkpoint=self.config.sam3_checkpoint)
            self.model.to(self.config.device).eval()
        return self.model

    def predict(self, x, batch_size=1, verbose=0):
        import torch
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError("TiSAM expects NHWC RGB input")
        model = self.build_model()
        device = next(model.parameters()).device
        mean = torch.tensor(self.config.data.mean, device=device)[None, :, None, None]
        std = torch.tensor(self.config.data.std, device=device)[None, :, None, None]
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                pixels = torch.from_numpy(x[start:start + batch_size]).permute(0, 3, 1, 2).to(device)
                logits = model((pixels / 255.0 - mean) / std)
                # Native [void, tissue...] -> CODAvision [tissue..., black].
                logits = torch.cat((logits[:, 1:], logits[:, :1]), dim=1)
                outputs.append(logits.permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(outputs)

    def save(self, path):
        from tisam.checkpointing.weights import export_weights, model_metadata
        export_weights(self.build_model(), path, metadata=model_metadata(
            self.config.model, class_names=self.config.data.class_names,
            mean=self.config.data.mean, std=self.config.data.std,
        ))

    def load(self, path):
        import json
        from tisam.checkpointing.weights import apply_weights, config_from_metadata, read_checkpoint
        state, metadata = read_checkpoint(Path(path))
        architecture = config_from_metadata(metadata)
        if architecture != self.config.model:
            raise ValueError("TiSAM checkpoint architecture disagrees with saved model settings")
        for key, expected in (("class_names", self.config.data.class_names),
                              ("mean", self.config.data.mean), ("std", self.config.data.std)):
            if key not in metadata or list(json.loads(metadata[key])) != list(expected):
                raise ValueError(f"TiSAM checkpoint {key} disagrees with saved model settings")
        apply_weights(self.build_model(), state)

    load_weights = load


def train_tisam(model_path, retrain_model=False, seed=None):
    """Train native TiSAM on the normal nesting-aware CODAvision tiles."""
    import shutil
    from base.data.loaders import load_model_metadata
    from base.models.utils import save_model_metadata
    from tisam.training.runner import train

    root = Path(model_path).resolve()
    best = root / "best_model_TiSAM.safetensors"
    if best.is_file() and not retrain_model:
        return
    metadata = load_model_metadata(str(root))
    config = native_config(metadata, options_from_metadata(metadata), root)
    if metadata.get("tile_format", "tif") not in ("tif", "png"):
        raise ValueError("TiSAM training requires TIFF or PNG tiles")
    if seed is not None:
        config.seed = seed
    # Regenerated tiles must not resume optimizer state from the previous dataset.
    # Keep previous native runs intact and use a fresh directory for retraining.
    if retrain_model and (root / config.name).exists():
        from datetime import datetime
        config.name += "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_model_metadata(str(root), {"framework": "pytorch"})
    train(config, resume=not retrain_model)
    source = config.out_dir / config.name / "checkpoints" / "best.safetensors"
    if not source.is_file():
        raise FileNotFoundError(f"TiSAM training did not produce {source}")
    temporary = best.with_suffix(".partial")
    shutil.copyfile(source, temporary)
    temporary.replace(best)
    save_model_metadata(str(root), {"training_completed": True})
