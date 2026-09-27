import pytest
import torch
import yaml

from carparts.utils import apply_overrides, load_config, pick_device, seed_everything


def test_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 0\ntrain:\n  lr: 1.0e-4\n  epochs: 10\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\ntrain:\n  epochs: 40\n")
    assert load_config(tmp_path / "child.yaml") == {"seed": 0, "train": {"lr": 1.0e-4, "epochs": 40}}


def test_yaml_scientific_notation_trap():
    # documents why configs write 1.0e-4: plain PyYAML returns a string here
    assert yaml.safe_load("lr: 1e-4")["lr"] == "1e-4"
    assert yaml.safe_load("lr: 1.0e-4")["lr"] == pytest.approx(1e-4)


def test_overrides_parse_types():
    cfg = apply_overrides(
        {"train": {"lr": 1.0e-4}},
        ["train.lr=3e-4", "train.amp=false", "model.name=segformer_b0", "data.scale=[0.5, 1.0]"],
    )
    assert isinstance(cfg["train"]["lr"], float) and cfg["train"]["lr"] == pytest.approx(3e-4)
    assert cfg["train"]["amp"] is False
    assert cfg["model"]["name"] == "segformer_b0"
    assert cfg["data"]["scale"] == [0.5, 1.0]


def test_bad_override_raises():
    with pytest.raises(ValueError):
        apply_overrides({}, ["train.lr"])


def test_seed_is_repeatable():
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_explicit_cpu_device():
    assert pick_device("cpu").type == "cpu"