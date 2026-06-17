"""Shape contracts and gradient checks for WaveUNet1D."""
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.waveunet1d import WaveUNet1D
from models.unet_registry import build_unet_variant
from train.unet_losses import build_loss_fn

B, T = 2, 1024


@pytest.mark.parametrize("bottleneck,dilation_rates", [
    ("standard", [1]),
    ("dilated", [1, 2, 4, 8]),
    ("attention_tcn", [1]),
])
def test_output_shape(bottleneck, dilation_rates):
    model = WaveUNet1D(signal_len=T, base_channels=16, depth=4,
                       bottleneck=bottleneck, dilation_rates=dilation_rates)
    model.eval()
    noisy = torch.randn(B, T)
    with torch.no_grad():
        out = model(noisy.unsqueeze(1))
    pred_noise = out["pred_noise"]
    assert pred_noise.shape == (B, 1, T), f"Expected ({B},1,{T}), got {pred_noise.shape}"


def test_residual_output_shape():
    model = WaveUNet1D(signal_len=T, base_channels=16, depth=4)
    model.eval()
    noisy = torch.randn(B, T)
    with torch.no_grad():
        out = model(noisy.unsqueeze(1))
        out_wave = noisy - out["pred_noise"].squeeze(1)
    assert out_wave.shape == (B, T)


def test_identity_safe_at_init():
    """Near-zero initial weights → pred_noise ≈ 0 → out_wave ≈ noisy."""
    model = WaveUNet1D(signal_len=T, base_channels=16, depth=4)
    for p in model.out_conv.parameters():
        torch.nn.init.zeros_(p)
    model.eval()
    noisy = torch.randn(B, T)
    with torch.no_grad():
        out = model(noisy.unsqueeze(1))
        out_wave = noisy - out["pred_noise"].squeeze(1)
    err = (out_wave - noisy).abs().max().item()
    assert err < 1e-4, f"Identity deviation {err:.2e} > 1e-4"


def test_one_step_finite_gradients():
    model = WaveUNet1D(signal_len=T, base_channels=16, depth=4, bottleneck="dilated",
                       dilation_rates=[1, 2, 4, 8])
    model.train()
    loss_fn = build_loss_fn("waveform_time_mrstft", time_loss_weight=1.0, mrstft_loss_weight=0.05)
    noisy = torch.randn(B, T)
    clean = torch.randn(B, T)
    out = model(noisy.unsqueeze(1))
    out["out_wave"] = noisy - out["pred_noise"].squeeze(1)
    total, components = loss_fn(out, clean, clean_spec=None)
    total.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


def test_registry_waveunet1d_shape():
    cfg = {
        "architecture": "waveunet1d",
        "signal_len": T,
        "base_channels": 16,
        "depth": 4,
        "bottleneck": "dilated",
        "dilation_rates": [1, 2, 4, 8],
    }
    model = build_unet_variant(cfg)
    model.eval()
    noisy = torch.randn(B, T)
    with torch.no_grad():
        out = model(noisy.unsqueeze(1))
        out_wave = noisy - out["pred_noise"].squeeze(1)
    assert out_wave.shape == (B, T)


def test_snr_loss_finite_gradients():
    model = WaveUNet1D(signal_len=T, base_channels=16, depth=4, bottleneck="dilated")
    model.train()
    loss_fn = build_loss_fn("waveform_time_mrstft_snr", time_loss_weight=1.0,
                            mrstft_loss_weight=0.05, snr_loss_weight=0.01)
    noisy = torch.randn(B, T)
    clean = torch.randn(B, T)
    out = model(noisy.unsqueeze(1))
    out["out_wave"] = noisy - out["pred_noise"].squeeze(1)
    total, _ = loss_fn(out, clean, clean_spec=None)
    assert torch.isfinite(total), "Loss is not finite"
    total.backward()