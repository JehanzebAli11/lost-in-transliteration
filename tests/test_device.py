"""Device choice for Phase 2 training and detector/classifier eval: auto picks cuda -> mps -> cpu.

bf16 stays CUDA-only; MPS (Apple silicon) runs fp32.
"""
import pytest
import torch

from src import eval_classifiers as ec, train_detector as td


def _avail(monkeypatch, cuda: bool, mps: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize("pick", [td.pick_device, ec.pick_device], ids=["train", "eval"])
@pytest.mark.parametrize("cuda,mps,want", [
    (True, True, "cuda"),
    (True, False, "cuda"),
    (False, True, "mps"),
    (False, False, "cpu"),
])
def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch, pick, cuda, mps, want):
    _avail(monkeypatch, cuda, mps)
    assert pick("auto") == want


@pytest.mark.parametrize("pick", [td.pick_device, ec.pick_device], ids=["train", "eval"])
@pytest.mark.parametrize("pref", ["cpu", "cuda", "mps"])
def test_explicit_device_is_kept(monkeypatch, pick, pref):
    _avail(monkeypatch, False, False)
    assert pick(pref) == pref


@pytest.mark.parametrize("device,cfg_bf16,supported,want", [
    ("cuda", True, True, True),
    ("cuda", True, False, False),
    ("cuda", False, True, False),
    ("mps", True, True, False),
    ("cpu", True, True, False),
])
def test_bf16_is_cuda_only(monkeypatch, device, cfg_bf16, supported, want):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: supported)
    assert td.use_bf16(device, {"bf16": cfg_bf16}) is want


def test_bf16_defaults_on_for_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert td.use_bf16("cuda", {}) is True
