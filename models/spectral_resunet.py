"""Spectral ResU-Net with residual conv blocks, optional dilated bottleneck and skip attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    g = groups
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, norm: str = "group", groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.norm1 = _gn(cout, groups) if norm == "group" else nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = _gn(cout, groups) if norm == "group" else nn.BatchNorm2d(cout)
        self.proj = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        y = F.leaky_relu(self.norm1(self.conv1(x)), 0.1, inplace=True)
        y = self.norm2(self.conv2(y))
        return F.leaky_relu(y + residual, 0.1, inplace=True)


class DilatedBottleneck2D(nn.Module):
    def __init__(self, channels: int, dilation_rates=(1, 2, 4), norm: str = "group"):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, d, dilation=d, bias=False),
                _gn(channels) if norm == "group" else nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.1, inplace=True),
            )
            for d in dilation_rates
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class AttentionGate(nn.Module):
    """Attention gate applied to encoder skip connections."""
    def __init__(self, gate_ch: int, feat_ch: int):
        super().__init__()
        self.Wg = nn.Conv2d(gate_ch, feat_ch, 1, bias=False)
        self.Wx = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        self.psi = nn.Conv2d(feat_ch, 1, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.relu(self.Wg(g) + self.Wx(x), inplace=True)))
        return x * alpha


class SpectralResUnet(nn.Module):
    """
    2-D spectral U-Net with residual conv blocks.

    Supports:
      - standard / dilated bottleneck
      - attention gates on skip connections
      - isotropic / freq_only_first2 pooling (mirrored in decoder)
      - mask_scaled_sigmoid and mask_softplus output modes
    """
    def __init__(
        self,
        input_shape: tuple,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        pooling_mode: str = "isotropic",
        output_mode: str = "mask_scaled_sigmoid",
        mask_max: float = 3.0,
        softplus_max: float = 3.0,
        bottleneck: str = "standard",
        dilation_rates=(1, 2, 4),
        skip_attention: bool = False,
        norm: str = "group",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_mode = output_mode
        self.mask_max = mask_max
        self.softplus_max = softplus_max
        self.depth = depth

        def _pool_kernel(i):
            if pooling_mode == "isotropic":
                return (2, 2)
            elif pooling_mode == "freq_only_first2":
                return (2, 1) if i < 2 else (2, 2)
            return (2, 1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        enc_ch = []
        for i in range(depth):
            nxt = base_channels * (2 ** i)
            self.enc_blocks.append(ResConvBlock(ch, nxt, norm=norm))
            self.pools.append(nn.MaxPool2d(_pool_kernel(i)))
            enc_ch.append(nxt)
            ch = nxt

        # Bottleneck
        bott = base_channels * (2 ** depth)
        self.bott_main = ResConvBlock(ch, bott, norm=norm)
        self.bott_extra: nn.Module
        if bottleneck == "dilated":
            self.bott_extra = DilatedBottleneck2D(bott, dilation_rates, norm=norm)
        else:
            self.bott_extra = nn.Identity()
        ch = bott

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.att_gates: nn.ModuleList | None = nn.ModuleList() if skip_attention else None
        for i in range(depth - 1, -1, -1):
            prev = enc_ch[i]
            k = _pool_kernel(i)
            self.up_convs.append(nn.ConvTranspose2d(ch, prev, kernel_size=k, stride=k))
            self.dec_blocks.append(ResConvBlock(prev * 2, prev, norm=norm))
            if skip_attention:
                self.att_gates.append(AttentionGate(ch, prev))
            ch = prev

        out_ch = 2 if output_mode in ("complex_crm", "complex_stft") else 1
        self.out_conv = nn.Conv2d(ch, out_ch, 1)

    def forward(
        self,
        x: torch.Tensor,
        noisy_mag: torch.Tensor | None = None,
        noisy_spec: torch.Tensor | None = None,
    ) -> dict:
        skips = []
        out = x
        for i in range(self.depth):
            out = self.enc_blocks[i](out)
            skips.append(out)
            out = self.pools[i](out)

        out = self.bott_main(out)
        out = self.bott_extra(out)

        for i in range(self.depth):
            gate = out
            skip = skips[-(i + 1)]
            up = self.up_convs[i](out)
            if up.shape[-2:] != skip.shape[-2:]:
                up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.att_gates is not None:
                skip = self.att_gates[i](skip, gate)
            out = torch.cat([up, skip], dim=1)
            out = self.dec_blocks[i](out)

        raw = self.out_conv(out)
        if raw.shape[-2:] != self.input_shape:
            raw = F.interpolate(raw, size=self.input_shape, mode="bilinear", align_corners=False)

        if self.output_mode == "mask_scaled_sigmoid":
            mask = self.mask_max * torch.sigmoid(raw)
            out_mag = mask * noisy_mag if noisy_mag is not None else None
            return {"out_mag": out_mag, "mask": mask}

        elif self.output_mode == "mask_softplus":
            mask = torch.clamp(F.softplus(raw), max=self.softplus_max)
            out_mag = mask * noisy_mag if noisy_mag is not None else None
            return {"out_mag": out_mag, "mask": mask}

        else:
            return {"out_mag": F.softplus(raw)}