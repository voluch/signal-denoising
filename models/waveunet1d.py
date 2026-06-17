"""1D Wave-U-Net for time-domain residual noise prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn1d(channels: int, groups: int = 8) -> nn.GroupNorm:
    g = groups
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResConvBlock1D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 3, 1, 1, bias=False),
            _gn1d(cout),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(cout, cout, 3, 1, 1, bias=False),
            _gn1d(cout),
        )
        self.proj = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.net(x) + self.proj(x), 0.1, inplace=True)


class DilatedBottleneck1D(nn.Module):
    def __init__(self, channels: int, dilation_rates=(1, 2, 4, 8)):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, 1, d, dilation=d, bias=False),
                _gn1d(channels),
                nn.LeakyReLU(0.1, inplace=True),
            )
            for d in dilation_rates
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class TCNBottleneck(nn.Module):
    """Small bottleneck self-attention over time dimension."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = _gn1d(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        xn = self.norm(x).permute(0, 2, 1)      # [B, T, C]
        attn_out, _ = self.attn(xn, xn, xn)
        return x + attn_out.permute(0, 2, 1)    # residual


class WaveUNet1D(nn.Module):
    """
    1D Wave-U-Net that predicts residual noise.

    Usage:
        pred_noise = model(noisy[:, None, :]).squeeze(1)
        out_wave = noisy - pred_noise

    Architecture:
        - Encoder: stride-2 Conv1d blocks (ResConvBlock1D + downsample)
        - Bottleneck: standard / dilated / attention_tcn
        - Decoder: ConvTranspose1d + skip concat + ResConvBlock1D
        - Output: 1-channel noise prediction
    """

    def __init__(
        self,
        signal_len: int = 1024,
        base_channels: int = 32,
        depth: int = 4,
        bottleneck: str = "standard",
        dilation_rates=(1, 2, 4, 8),
    ):
        super().__init__()
        self.signal_len = signal_len
        self.depth = depth
        self.bottleneck_type = bottleneck

        self.enc_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        enc_ch = []
        ch = 1
        for i in range(depth):
            nxt = base_channels * (2 ** i)
            self.enc_blocks.append(ResConvBlock1D(ch, nxt))
            # stride-2 downsampling
            self.down_convs.append(nn.Conv1d(nxt, nxt, 4, stride=2, padding=1, bias=False))
            enc_ch.append(nxt)
            ch = nxt

        bott = base_channels * (2 ** depth)
        self.bott_block = ResConvBlock1D(ch, bott)
        if bottleneck == "dilated":
            self.bott_extra: nn.Module = DilatedBottleneck1D(bott, dilation_rates)
        elif bottleneck == "attention_tcn":
            self.bott_extra = TCNBottleneck(bott)
        else:
            self.bott_extra = nn.Identity()
        ch = bott

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            prev = enc_ch[i]
            self.up_convs.append(nn.ConvTranspose1d(ch, prev, 4, stride=2, padding=1))
            self.dec_blocks.append(ResConvBlock1D(prev * 2, prev))
            ch = prev

        self.out_conv = nn.Conv1d(ch, 1, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        """x: [B, 1, T] → dict with 'pred_noise' [B, 1, T]."""
        skips = []
        out = x
        for i in range(self.depth):
            out = self.enc_blocks[i](out)
            skips.append(out)
            out = self.down_convs[i](out)

        out = self.bott_block(out)
        out = self.bott_extra(out)

        for i in range(self.depth):
            skip = skips[-(i + 1)]
            up = self.up_convs[i](out)
            if up.shape[-1] != skip.shape[-1]:
                up = F.interpolate(up, size=skip.shape[-1], mode="linear", align_corners=False)
            out = torch.cat([up, skip], dim=1)
            out = self.dec_blocks[i](out)

        pred_noise = self.out_conv(out)          # [B, 1, T]
        if pred_noise.shape[-1] != self.signal_len:
            pred_noise = F.interpolate(pred_noise, size=self.signal_len, mode="linear", align_corners=False)
        return {"pred_noise": pred_noise}
