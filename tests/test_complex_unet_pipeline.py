"""
Complex CRM pipeline smoke tests.

1. Identity reconstruction test — CRM initialized to (1+0j) should pass noisy through.
2. Zero-output prevention — zero final layer → identity output, not silence.
3. Oracle CRM test — target CRM reconstructs clean with high SNR.
4. Loss-gradient test — complex_time_mrstft produces finite gradients.
5. Shape test — out_spec [B,F,T] and out_wave [B,signal_len].
"""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.stft_projector import STFTProjector
from models.unet_registry import build_unet_variant
from train.unet_losses import build_loss_fn

B, T = 2, 1024
PROJ = STFTProjector(nperseg=128, hop_length=32, signal_len=T)
SPEC_SHAPE = PROJ.spec_shape()

CRM_CFG = {
    "architecture": "complex_crm_unet",
    "input_domain": "real_imag_mag",
    "output_mode": "complex_crm_identity_residual",
    "crm_scale": 0.5,
    "pooling_mode": "isotropic",
}


def _batch():
    noisy = torch.randn(B, T)
    clean = torch.randn(B, T) * 0.3
    spec_n = PROJ.stft(noisy)
    mag_n = spec_n.abs().unsqueeze(1)
    x_in = torch.stack([spec_n.real, spec_n.imag, mag_n.squeeze(1)], dim=1)
    return noisy, clean, spec_n, mag_n, x_in


def test_shape_out_spec_and_wave():
    model = build_unet_variant(CRM_CFG, SPEC_SHAPE)
    model.eval()
    noisy, clean, spec_n, mag_n, x_in = _batch()
    with torch.no_grad():
        out = model(x_in, noisy_spec=spec_n, noisy_mag=mag_n)
        out_wave = PROJ.istft(out["out_spec"], T)
    F_bins, T_frames = SPEC_SHAPE
    assert out["out_spec"].shape == (B, F_bins, T_frames), \
        f"out_spec shape {out['out_spec'].shape}"
    assert out_wave.shape == (B, T), f"out_wave shape {out_wave.shape}"


def test_identity_crm_passthrough():
    """With near-zero raw output (after init), CRM ≈ 1+0j → output ≈ noisy input."""
    model = build_unet_variant(CRM_CFG, SPEC_SHAPE)
    # Zero all weights in the final output conv → raw ≈ 0 → tanh(0)=0 → CRM=(1+0j)
    for name, p in model.named_parameters():
        if "out_conv" in name:
            torch.nn.init.zeros_(p)
    model.eval()
    noisy, _, spec_n, mag_n, x_in = _batch()
    with torch.no_grad():
        out = model(x_in, noisy_spec=spec_n, noisy_mag=mag_n)
        out_wave = PROJ.istft(out["out_spec"], T)
    noisy_reconstructed = PROJ.istft(spec_n, T)
    err = (out_wave - noisy_reconstructed).abs().max().item()
    assert err < 1e-3, f"Identity CRM deviation {err:.4e} > 1e-3"


def test_zero_output_prevention():
    """With zero final layer, model must NOT output silence (out_wave ≠ 0)."""
    model = build_unet_variant(CRM_CFG, SPEC_SHAPE)
    for name, p in model.named_parameters():
        if "out_conv" in name:
            torch.nn.init.zeros_(p)
    model.eval()
    noisy, _, spec_n, mag_n, x_in = _batch()
    with torch.no_grad():
        out = model(x_in, noisy_spec=spec_n, noisy_mag=mag_n)
        out_wave = PROJ.istft(out["out_spec"], T)
    assert out_wave.abs().max().item() > 1e-6, "Output is silence — identity path broken"


def test_oracle_crm_snr():
    """Oracle CRM = clean_spec / (noisy_spec + eps) reconstructs clean with high SNR."""
    noisy = torch.randn(B, T)
    clean = noisy * 0.8 + 0.2 * torch.randn(B, T)  # correlated so CRM is informative
    spec_n = PROJ.stft(noisy)
    spec_c = PROJ.stft(clean)
    eps = 1e-8
    oracle_crm = spec_c / (spec_n + eps)
    out_spec = oracle_crm * spec_n
    recon = PROJ.istft(out_spec, T)
    # SNR of reconstruction vs clean
    noise_pwr = ((recon - clean) ** 2).mean()
    sig_pwr = (clean ** 2).mean() + 1e-12
    snr_db = 10 * torch.log10(sig_pwr / noise_pwr).item()
    assert snr_db > 10.0, f"Oracle CRM SNR {snr_db:.1f} dB < 10 dB"


def test_loss_gradient_finite():
    """complex_time_mrstft loss must produce finite gradients."""
    model = build_unet_variant(CRM_CFG, SPEC_SHAPE)
    model.train()
    loss_fn = build_loss_fn("complex_time_mrstft", time_loss_weight=0.1, mrstft_loss_weight=0.01, snr_loss_weight=0.0)
    noisy, clean, spec_n, mag_n, x_in = _batch()
    out = model(x_in, noisy_spec=spec_n, noisy_mag=mag_n)
    out_wave = PROJ.istft(out["out_spec"], T)
    out["out_wave"] = out_wave
    clean_spec = PROJ.stft(clean)
    total, _ = loss_fn(out, clean, clean_spec)
    total.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


def test_complex_stft_direct_shape():
    cfg = {
        "architecture": "complex_stft_unet",
        "input_domain": "real_imag_mag",
        "output_mode": "complex_stft",
        "pooling_mode": "isotropic",
    }
    model = build_unet_variant(cfg, SPEC_SHAPE)
    model.eval()
    noisy, _, spec_n, mag_n, x_in = _batch()
    with torch.no_grad():
        out = model(x_in, noisy_spec=spec_n, noisy_mag=mag_n)
        out_wave = PROJ.istft(out["out_spec"], T)
    assert out_wave.shape == (B, T)
    assert torch.isfinite(out_wave).all()