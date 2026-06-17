"""Shared STFT/iSTFT module used by all spectral U-Net variants."""
import torch
import torch.nn as nn


class STFTProjector(nn.Module):
    def __init__(self, nperseg: int = 128, hop_length: int = 32, signal_len: int = 1024):
        super().__init__()
        self.nperseg = nperseg
        self.hop_length = hop_length
        self.signal_len = signal_len

    def _win(self, device, dtype):
        return torch.hann_window(self.nperseg, periodic=True, device=device, dtype=dtype)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] → complex [B, F, TT] (onesided, center-padded)."""
        win = self._win(x.device, x.dtype)
        return torch.stft(
            x, n_fft=self.nperseg, hop_length=self.hop_length,
            win_length=self.nperseg, window=win,
            center=True, pad_mode="reflect", onesided=True, return_complex=True,
        )

    def istft(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """complex [B, F, TT] → [B, T]."""
        win = self._win(spec.device, spec.real.dtype)
        return torch.istft(
            spec, n_fft=self.nperseg, hop_length=self.hop_length,
            win_length=self.nperseg, window=win,
            center=True, onesided=True, length=length or self.signal_len,
        )

    def mag_phase(self, x: torch.Tensor):
        """[B, T] → mag [B, 1, F, TT], phase [B, F, TT] (unit complex)."""
        spec = self.stft(x)
        mag = spec.abs().unsqueeze(1)
        phase = spec / (spec.abs() + 1e-8)
        return mag, phase

    def spec_shape(self, device="cpu") -> tuple[int, int]:
        dummy = torch.zeros(1, self.signal_len, device=device)
        s = self.stft(dummy)
        return (s.shape[1], s.shape[2])