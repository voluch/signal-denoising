"""Complex spectral U-Net variants with identity-safe CRM initialization."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectral_resunet import ResConvBlock, _gn


def _pool_kernel(i: int, pooling_mode: str) -> tuple:
    if pooling_mode == "isotropic":
        return (2, 2)
    elif pooling_mode == "freq_only_first2":
        return (2, 1) if i < 2 else (2, 2)
    return (2, 1)


class _UNetBackbone(nn.Module):
    """Shared encoder-decoder backbone for complex U-Net variants."""

    def __init__(self, input_shape, in_channels, out_channels, base_channels, depth, pooling_mode, norm="group"):
        super().__init__()
        self.input_shape = input_shape
        self.depth = depth

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        enc_ch = []
        for i in range(depth):
            nxt = base_channels * (2 ** i)
            self.enc_blocks.append(ResConvBlock(ch, nxt, norm=norm))
            self.pools.append(nn.MaxPool2d(_pool_kernel(i, pooling_mode)))
            enc_ch.append(nxt)
            ch = nxt

        bott = base_channels * (2 ** depth)
        self.bottleneck = ResConvBlock(ch, bott, norm=norm)
        ch = bott

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            prev = enc_ch[i]
            k = _pool_kernel(i, pooling_mode)
            self.up_convs.append(nn.ConvTranspose2d(ch, prev, kernel_size=k, stride=k))
            self.dec_blocks.append(ResConvBlock(prev * 2, prev, norm=norm))
            ch = prev

        self.out_conv = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x
        for i in range(self.depth):
            out = self.enc_blocks[i](out)
            skips.append(out)
            out = self.pools[i](out)

        out = self.bottleneck(out)

        for i in range(self.depth):
            skip = skips[-(i + 1)]
            up = self.up_convs[i](out)
            if up.shape[-2:] != skip.shape[-2:]:
                up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = torch.cat([up, skip], dim=1)
            out = self.dec_blocks[i](out)

        raw = self.out_conv(out)
        if raw.shape[-2:] != self.input_shape:
            raw = F.interpolate(raw, size=self.input_shape, mode="bilinear", align_corners=False)
        return raw


class ComplexCRMUnet(nn.Module):
    """
    Complex Ratio Mask U-Net with bounded residual CRM around identity.

    Safe initialization: if raw output → 0 at init, CRM → (1+0j), identity pass-through.

    Formula:
        mr = 1.0 + crm_scale * tanh(raw_real)
        mi = crm_scale * tanh(raw_imag)
        crm = complex(mr, mi)
        out_spec = crm * noisy_spec
    """

    def __init__(
        self,
        input_shape: tuple,
        in_channels: int = 3,
        base_channels: int = 32,
        depth: int = 3,
        pooling_mode: str = "isotropic",
        crm_scale: float = 0.5,
        norm: str = "group",
    ):
        super().__init__()
        self.crm_scale = crm_scale
        self.backbone = _UNetBackbone(
            input_shape, in_channels, 2, base_channels, depth, pooling_mode, norm
        )

    def forward(
        self,
        x: torch.Tensor,
        noisy_spec: torch.Tensor | None = None,
        noisy_mag: torch.Tensor | None = None,
    ) -> dict:
        raw = self.backbone(x)  # [B, 2, F, T]
        mr = 1.0 + self.crm_scale * torch.tanh(raw[:, 0])   # [B, F, T]
        mi = self.crm_scale * torch.tanh(raw[:, 1])           # [B, F, T]
        crm = torch.complex(mr, mi)

        result = {"crm": crm}
        if noisy_spec is not None:
            out_spec = crm * noisy_spec
            result["out_spec"] = out_spec
        return result


class ComplexSTFTUnet(nn.Module):
    """
    Predict clean complex STFT directly (real and imaginary channels).

    Unlike CRM, this does not apply a mask but directly predicts the target spectrum.
    """

    def __init__(
        self,
        input_shape: tuple,
        in_channels: int = 3,
        base_channels: int = 32,
        depth: int = 3,
        pooling_mode: str = "isotropic",
        norm: str = "group",
    ):
        super().__init__()
        self.backbone = _UNetBackbone(
            input_shape, in_channels, 2, base_channels, depth, pooling_mode, norm
        )

    def forward(
        self,
        x: torch.Tensor,
        noisy_spec: torch.Tensor | None = None,
        noisy_mag: torch.Tensor | None = None,
    ) -> dict:
        raw = self.backbone(x)  # [B, 2, F, T]
        out_spec = torch.complex(raw[:, 0], raw[:, 1])  # [B, F, T]
        return {"out_spec": out_spec}
