"""Shape contracts and gradient checks for all spectral U-Net variants."""
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.stft_projector import STFTProjector
from models.unet_registry import build_unet_variant, param_count

B, T = 2, 1024
PROJ = STFTProjector(nperseg=128, hop_length=32, signal_len=T)
SPEC_SHAPE = PROJ.spec_shape()


def _make_batch(b=B):
    noisy = torch.randn(b, T)
    clean = torch.randn(b, T)
    spec = PROJ.stft(noisy)
    mag = spec.abs().unsqueeze(1)
    return noisy, clean, spec, mag


def _forward(cfg: dict):
    noisy, clean, spec, mag = _make_batch()
    model = build_unet_variant(cfg, input_shape=SPEC_SHAPE)
    model.eval()
    with torch.no_grad():
        if cfg.get("architecture") == "waveunet1d":
            out = model(noisy.unsqueeze(1))
            out["out_wave"] = noisy - out["pred_noise"].squeeze(1)
        else:
            x_in = mag
            if cfg.get("input_domain") == "real_imag_mag":
                x_in = torch.stack([spec.real, spec.imag, mag.squeeze(1)], dim=1)
            out = model(x_in, noisy_mag=mag, noisy_spec=spec)
            if "out_wave" not in out:
                if "out_spec" in out:
                    os = out["out_spec"]
                    out["out_wave"] = PROJ.istft(os, T)
                elif "out_mag" in out:
                    phase = spec / (spec.abs() + 1e-8)
                    out["out_wave"] = PROJ.istft(out["out_mag"].squeeze(1) * phase, T)
    return out


@pytest.mark.parametrize("cfg", [
    {"architecture": "spectral_unet", "input_domain": "mag",
     "output_mode": "mask_scaled_sigmoid", "mask_max": 3.0},
    {"architecture": "spectral_unet", "input_domain": "mag",
     "output_mode": "mask_softplus", "softplus_max": 3.0},
    {"architecture": "spectral_resunet", "input_domain": "mag",
     "output_mode": "mask_scaled_sigmoid", "mask_max": 3.0},
    {"architecture": "spectral_resunet", "input_domain": "mag",
     "output_mode": "mask_scaled_sigmoid", "mask_max": 3.0,
     "bottleneck": "dilated", "dilation_rates": [1, 2, 4]},
    {"architecture": "spectral_resunet", "input_domain": "mag",
     "output_mode": "mask_scaled_sigmoid", "mask_max": 3.0, "skip_attention": True},
    {"architecture": "spectral_resunet", "input_domain": "mag",
     "output_mode": "mask_scaled_sigmoid", "mask_max": 3.0,
     "pooling_mode": "freq_only_first2"},
])
def test_spectral_out_wave_shape(cfg):
    out = _forward(cfg)
    assert "out_wave" in out, "out_wave missing"
    assert out["out_wave"].shape == (B, T), f"Expected ({B},{T}), got {out['out_wave'].shape}"


@pytest.mark.parametrize("cfg", [
    {"architecture": "spectral_unet", "input_domain": "mag",
     "output_mode": "mask_scaled_sigmoid", "mask_max": 3.0},
    {"architecture": "spectral_resunet", "input_domain": "mag",
     "output_mode": "mask_scaled_sigmoid", "mask_max": 3.0},
])
def test_one_step_finite_gradients(cfg):
    noisy, clean, spec, mag = _make_batch()
    model = build_unet_variant(cfg, input_shape=SPEC_SHAPE)
    model.train()
    x_in = mag
    out = model(x_in, noisy_mag=mag, noisy_spec=spec)
    phase = spec / (spec.abs() + 1e-8)
    out_wave = PROJ.istft(out["out_mag"].squeeze(1) * phase, T)
    clean_spec = PROJ.stft(clean)
    loss = torch.nn.functional.mse_loss(out["out_mag"], clean_spec.abs().unsqueeze(1))
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


def test_model_config_roundtrip():
    cfg = {"architecture": "spectral_resunet", "input_domain": "mag",
           "output_mode": "mask_scaled_sigmoid", "mask_max": 3.0,
           "base_channels": 32, "depth": 3}
    m1 = build_unet_variant(cfg, SPEC_SHAPE)
    m2 = build_unet_variant(cfg, SPEC_SHAPE)
    assert param_count(m1) == param_count(m2)


def test_param_count_logged():
    cfg = {"architecture": "spectral_unet"}
    m = build_unet_variant(cfg, SPEC_SHAPE)
    assert param_count(m) > 0