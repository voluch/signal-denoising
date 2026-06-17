"""Tests for STFTProjector round-trip fidelity and shape contracts."""
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.stft_projector import STFTProjector


@pytest.fixture
def proj():
    return STFTProjector(nperseg=128, hop_length=32, signal_len=1024)


def test_stft_output_shape(proj):
    x = torch.randn(4, 1024)
    spec = proj.stft(x)
    assert spec.shape[0] == 4
    assert spec.shape[1] == 65        # nperseg/2 + 1
    assert spec.is_complex()


def test_istft_roundtrip_length(proj):
    x = torch.randn(4, 1024)
    spec = proj.stft(x)
    x_hat = proj.istft(spec, length=1024)
    assert x_hat.shape == (4, 1024)


def test_istft_roundtrip_error(proj):
    x = torch.randn(4, 1024)
    spec = proj.stft(x)
    x_hat = proj.istft(spec, length=1024)
    err = (x_hat - x).abs().max().item()
    assert err < 1e-4, f"Max reconstruction error {err:.2e} > 1e-4"


def test_mag_phase_shapes(proj):
    x = torch.randn(4, 1024)
    mag, phase = proj.mag_phase(x)
    spec = proj.stft(x)
    assert mag.shape == (4, 1, spec.shape[1], spec.shape[2])
    assert phase.shape == spec.shape


def test_phase_unit_magnitude(proj):
    x = torch.randn(4, 1024)
    _, phase = proj.mag_phase(x)
    abs_phase = phase.abs()
    # All non-zero bins should have unit magnitude
    nonzero = abs_phase[abs_phase > 1e-6]
    assert torch.allclose(nonzero, torch.ones_like(nonzero), atol=1e-5)


def test_hop16_shape(proj):
    p16 = STFTProjector(nperseg=128, hop_length=16, signal_len=1024)
    x = torch.randn(2, 1024)
    spec = p16.stft(x)
    x_hat = p16.istft(spec, length=1024)
    assert x_hat.shape == (2, 1024)
    err = (x_hat - x).abs().max().item()
    assert err < 1e-4


def test_spec_shape_helper():
    proj = STFTProjector(nperseg=128, hop_length=32, signal_len=1024)
    F, T = proj.spec_shape()
    assert F == 65
    assert T > 0