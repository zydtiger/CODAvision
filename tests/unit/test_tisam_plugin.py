"""Plugin registration and the real boundary between CODAvision and TiSAM."""
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from base.config import ModelDefaults
from base.models.base import BaseSegmentationModelInterface, Framework
from base.models.backbones import model_call
from base.models.backbones_tisam import TiSAMSegmentationModel, native_config, parse_options


def test_registration_and_factory():
    assert "TiSAM" in ModelDefaults.PYTORCH_MODELS
    assert "TiSAM" not in ModelDefaults.TENSORFLOW_MODELS
    model = model_call("TiSAM", 32, 3)
    assert isinstance(model, BaseSegmentationModelInterface)
    assert model.framework == Framework.PYTORCH
    assert model.model is None  # No encoder downloads merely to select a model.
    with pytest.raises(ValueError, match="requires the pytorch"):
        model_call("TiSAM", 32, 3, framework="tensorflow")


def test_factory_raw_model(monkeypatch):
    raw = object()
    monkeypatch.setattr(TiSAMSegmentationModel, "build_model", lambda self: raw)
    assert model_call("TiSAM", 32, 3, wrap_with_adapter=False) is raw


def test_native_tiles_keep_class_ids_and_nesting(tmp_path):
    config = native_config(
        {"sxy": 512, "classNames": ["tumor", "whitespace", "black"], "batch_size": 2},
        {}, tmp_path,
    )
    assert config.data.class_names == ["void", "tumor", "whitespace"]
    source = config.data.train[0]
    assert source.path == tmp_path / "training"
    assert source.metadata.label_to_class_name == ["void", "tumor", "whitespace", "void"]
    assert source.metadata.supervision_ignored_raw_labels == [0]
    assert config.model.num_classes == 2
    assert config.batch_size == 2


@pytest.mark.parametrize("text", ["[]", "model: {num_classes: 4}", "training: {typo: 1}", "data: {train: []}"])
def test_invalid_settings(text):
    with pytest.raises(ValueError):
        parse_options(text)


def test_rgb_normalization_and_black_channel_order():
    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.inputs = []
        def forward(self, pixels):
            self.inputs.append(pixels.clone())
            return torch.stack((pixels[:, 0] * 0 + 9, pixels[:, 0] * 0 + 2, pixels[:, 0] * 0 + 4), dim=1)
    adapter = model_call("TiSAM", 4, 3, model_options={"training": {"device": "cpu"}})
    adapter.model = Native()
    rgb = np.full((2, 4, 4, 3), 255, dtype=np.float32)
    result = adapter.predict(rgb, batch_size=1)
    assert result.shape == (2, 4, 4, 3)
    np.testing.assert_array_equal(result[0, 0, 0], [2, 4, 9])
    np.testing.assert_allclose(adapter.model.inputs[0][0, :, 0, 0],
                               (1 - np.array(adapter.config.data.mean)) / np.array(adapter.config.data.std), rtol=1e-6)


def test_existing_gui_dropdown(qapp):
    from gui.components.ui_definitions import Ui_MainWindow
    from PySide6.QtWidgets import QMainWindow
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    assert ui.model_type_CB.findText("TiSAM") >= 0
    ui.model_type_CB.setCurrentText("TiSAM")
    assert not ui.tisam_options.isHidden()
    assert ui.tabWidget.indexOf(ui.tab_3) >= 0  # Original nesting tab retained.
    ui.model_type_CB.setCurrentText("UNet")
    assert ui.tisam_options.isHidden()
    window.close()


def test_classification_uses_plugin(tmp_path, monkeypatch):
    from base.image.classification import ImageClassifier
    import base.models.backbones as factory
    classifier = ImageClassifier.__new__(ImageClassifier)
    classifier.model_type = "TiSAM"
    classifier.model_path = str(tmp_path)
    classifier.image_size = 32
    classifier.class_names = ["tissue", "black"]
    classifier.model_data = {"classNames": classifier.class_names}
    model = Mock()
    call = Mock(return_value=model)
    monkeypatch.setattr(factory, "model_call", call)
    assert classifier._load_model() is model
    model.load.assert_called_once_with(str(tmp_path / "best_model_TiSAM.safetensors"))


def test_training_dispatch_and_checkpoint_publication(tmp_path, monkeypatch):
    from base.models.training import train_segmentation_model_cnns
    from base.models.utils import save_model_metadata
    import tisam.training.runner as runner
    save_model_metadata(str(tmp_path), {
        "model_type": "TiSAM", "classNames": ["tissue", "black"],
        "sxy": 32, "batch_size": 1, "nblack": 2, "nwhite": 1,
    })
    calls = []
    def train(config, **kwargs):
        calls.append(config)
        checkpoint = config.out_dir / config.name / "checkpoints" / "best.safetensors"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"test-checkpoint")
    monkeypatch.setattr(runner, "train", train)
    train_segmentation_model_cnns(str(tmp_path), seed=7)
    assert calls[0].seed == 7
    assert (tmp_path / "best_model_TiSAM.safetensors").read_bytes() == b"test-checkpoint"
    train_segmentation_model_cnns(str(tmp_path))
    assert len(calls) == 1
