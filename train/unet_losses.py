"""
Loss factory for U-Net iteration-2 experiments.

All composite losses log component values separately in a dict keyed by:
  loss_total, loss_mag, loss_time, loss_mrstft, loss_snr, loss_complex
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────

def negative_snr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    noise = pred - target
    sig_pwr = torch.mean(target ** 2, dim=-1) + eps
    noise_pwr = torch.mean(noise ** 2, dim=-1) + eps
    return -10.0 * torch.log10(sig_pwr / noise_pwr).mean()


def _stft_mag(x: torch.Tensor, n: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n, periodic=True, device=x.device, dtype=x.dtype)
    return torch.abs(torch.stft(x, n_fft=n, hop_length=hop, win_length=n,
                                window=win, center=True, return_complex=True))


def multi_res_stft_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    configs: tuple = ((32, 16), (64, 32), (16, 8)),
) -> torch.Tensor:
    total = x_hat.new_zeros(1).squeeze()
    for n, ov in configs:
        hop = n - ov
        S_hat = _stft_mag(x_hat, n, hop)
        S = _stft_mag(x, n, hop)
        l1 = torch.mean(torch.abs(torch.log1p(S_hat) - torch.log1p(S)))
        sc = (
            torch.linalg.norm(S_hat - S, ord="fro", dim=(1, 2))
            / (torch.linalg.norm(S, ord="fro", dim=(1, 2)) + 1e-12)
        ).mean()
        total = total + l1 + 0.5 * sc
    return total / len(configs)


# ── factory ───────────────────────────────────────────────────────────────────

SUPPORTED_PROFILES = {
    "mag",
    "mag_time",
    "time_mrstft",
    "mag_time_mrstft",
    "mag_time_mrstft_snr",
    "snr_aux",
    "complex_time",
    "complex_time_mrstft",
    "waveform_time_mrstft",
    "waveform_time_mrstft_snr",
}


def build_loss_fn(
    loss_profile: str,
    time_loss_weight: float = 0.03,
    mrstft_loss_weight: float = 0.01,
    snr_loss_weight: float = 0.01,
):
    """
    Return a loss callable with signature:

        total, components = fn(out_dict, clean_wave, clean_spec)

    out_dict must contain the keys consumed by the chosen profile:
        mag profiles      → out_dict["out_mag"] [B,1,F,T], out_dict["out_wave"] [B,T]
        complex profiles  → out_dict["out_spec"] complex [B,F,T], out_dict["out_wave"] [B,T]
        waveform profiles → out_dict["out_wave"] [B,T]

    clean_spec may be None for waveform-only profiles.
    """
    if loss_profile not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unknown loss_profile '{loss_profile}'. Supported: {sorted(SUPPORTED_PROFILES)}"
        )

    def _fn(
        out_dict: dict,
        clean_wave: torch.Tensor,
        clean_spec: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        components: dict[str, torch.Tensor] = {}
        total = clean_wave.new_zeros(1).squeeze()

        out_wave = out_dict.get("out_wave")
        out_mag = out_dict.get("out_mag")
        out_spec = out_dict.get("out_spec")
        clean_mag = clean_spec.abs().unsqueeze(1) if clean_spec is not None else None

        # ── magnitude spectral loss ──────────────────────────────────────────
        if loss_profile in ("mag", "mag_time", "mag_time_mrstft", "mag_time_mrstft_snr", "snr_aux"):
            components["loss_mag"] = F.mse_loss(out_mag, clean_mag)
            total = total + components["loss_mag"]

        # ── complex STFT loss ────────────────────────────────────────────────
        if loss_profile in ("complex_time", "complex_time_mrstft"):
            components["loss_complex"] = F.mse_loss(
                torch.view_as_real(out_spec),
                torch.view_as_real(clean_spec),
            )
            total = total + components["loss_complex"]

        # ── waveform-only profiles (no spectral base) ────────────────────────
        if loss_profile in ("waveform_time_mrstft", "waveform_time_mrstft_snr"):
            components["loss_time"] = F.mse_loss(out_wave, clean_wave)
            total = total + time_loss_weight * components["loss_time"]
            if mrstft_loss_weight > 0:
                components["loss_mrstft"] = multi_res_stft_loss(out_wave, clean_wave)
                total = total + mrstft_loss_weight * components["loss_mrstft"]
            if loss_profile == "waveform_time_mrstft_snr" and snr_loss_weight > 0:
                components["loss_snr"] = negative_snr_loss(out_wave, clean_wave)
                total = total + snr_loss_weight * components["loss_snr"]
            components["loss_total"] = total
            return total, components

        # ── time-domain add-on (shared by remaining profiles) ───────────────
        if loss_profile in (
            "mag_time", "mag_time_mrstft", "mag_time_mrstft_snr",
            "time_mrstft", "complex_time", "complex_time_mrstft", "snr_aux",
        ):
            if out_wave is not None and time_loss_weight > 0:
                components["loss_time"] = F.mse_loss(out_wave, clean_wave)
                total = total + time_loss_weight * components["loss_time"]

        # ── MR-STFT add-on ───────────────────────────────────────────────────
        if loss_profile in (
            "mag_time_mrstft", "mag_time_mrstft_snr",
            "time_mrstft", "complex_time_mrstft",
        ):
            if out_wave is not None and mrstft_loss_weight > 0:
                components["loss_mrstft"] = multi_res_stft_loss(out_wave, clean_wave)
                total = total + mrstft_loss_weight * components["loss_mrstft"]

        # ── SNR auxiliary loss ───────────────────────────────────────────────
        if loss_profile in ("mag_time_mrstft_snr", "snr_aux"):
            if out_wave is not None and snr_loss_weight > 0:
                components["loss_snr"] = negative_snr_loss(out_wave, clean_wave)
                total = total + snr_loss_weight * components["loss_snr"]

        components["loss_total"] = total
        return total, components

    return _fn