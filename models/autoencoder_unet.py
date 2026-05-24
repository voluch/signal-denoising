# models/autoencoder_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- helpers --------
def conv_block(cin, cout, k=3, s=1, p=1, use_dropout=False, dropout_p=0.0):
    """2x(Conv+BN+LeakyReLU) with optional Dropout."""
    layers = [
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(inplace=True),
    ]
    if use_dropout:
        layers.append(nn.Dropout2d(dropout_p))
    layers.extend([
        nn.Conv2d(cout, cout, k, 1, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(inplace=True),
    ])
    return nn.Sequential(*layers)

class SE(nn.Module):
    """Squeeze-and-Excitation for channel attention."""
    def __init__(self, c, r=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)

# -------- model --------
class UnetAutoencoder(nn.Module):
    """
    Highly parameterized U-Net for signal denoising experiments.
    Supports various pooling modes, output modes, and complex-valued data.
    """
    def __init__(
        self,
        input_shape,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        pooling_mode: str = "isotropic",       # isotropic | freq_only_first2 | time_preserve
        output_mode: str = "mask_sigmoid",     # mask_sigmoid | mask_scaled_sigmoid | mask_softplus | direct_mag | residual_mag | complex_crm | complex_stft
        mask_max: float = 1.0,
        softplus_max: float = 3.0,
        use_se: bool = True,
        use_dropout: bool = False,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.input_shape = input_shape  # (F, T)
        self.output_mode = output_mode
        self.mask_max = mask_max
        self.softplus_max = softplus_max
        self.depth = depth

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.se_blocks = nn.ModuleList() if use_se else None

        curr_channels = in_channels
        for i in range(depth):
            next_channels = base_channels * (2 ** i)
            self.enc_blocks.append(conv_block(curr_channels, next_channels, use_dropout=use_dropout, dropout_p=dropout_p))
            if use_se:
                self.se_blocks.append(SE(next_channels))
            
            # Resolve pooling
            if pooling_mode == "isotropic":
                kernel = (2, 2)
            elif pooling_mode == "freq_only_first2":
                kernel = (2, 1) if i < 2 else (2, 2)
            elif pooling_mode == "time_preserve":
                kernel = (2, 1)
            else:
                raise ValueError(f"Unknown pooling_mode: {pooling_mode}")
            
            self.pools.append(nn.MaxPool2d(kernel_size=kernel, stride=kernel))
            curr_channels = next_channels

        # Bottleneck
        bott_channels = base_channels * (2 ** depth)
        self.bottleneck = conv_block(curr_channels, bott_channels, use_dropout=use_dropout, dropout_p=dropout_p)
        self.bottleneck_se = SE(bott_channels) if use_se else nn.Identity()

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_se_blocks = nn.ModuleList() if use_se else None

        curr_channels = bott_channels
        for i in range(depth - 1, -1, -1):
            prev_channels = base_channels * (2 ** i)
            
            # Upsampling must mirror pooling
            if pooling_mode == "isotropic":
                kernel = (2, 2)
            elif pooling_mode == "freq_only_first2":
                kernel = (2, 1) if i < 2 else (2, 2)
            elif pooling_mode == "time_preserve":
                kernel = (2, 1)
            
            self.up_convs.append(nn.ConvTranspose2d(curr_channels, prev_channels, kernel_size=kernel, stride=kernel))
            self.dec_blocks.append(conv_block(prev_channels * 2, prev_channels, use_dropout=use_dropout, dropout_p=dropout_p))
            if use_se:
                self.dec_se_blocks.append(SE(prev_channels))
            curr_channels = prev_channels

        # Final output layer
        # complex_crm and complex_stft need 2 output channels (real/imag)
        actual_out_channels = 2 if output_mode in ["complex_crm", "complex_stft"] else out_channels
        self.out_conv = nn.Conv2d(curr_channels, actual_out_channels, kernel_size=1)
        self._final_interp = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x, noisy_mag=None, noisy_spec=None):
        """
        x: Input tensor (B, Cin, F, T)
        noisy_mag: (B, 1, F, T) optional noisy magnitude for masking
        noisy_spec: (B, F, T) complex optional noisy spectrogram for complex masking
        
        Returns:
            If complex_stft: Dictionary with 'out_spec' (complex)
            Else: Dictionary with 'out_mag' (and optionally 'mask')
        """
        # Encoder
        skip_connections = []
        out = x
        for i in range(self.depth):
            out = self.enc_blocks[i](out)
            if self.se_blocks:
                out = self.se_blocks[i](out)
            skip_connections.append(out)
            out = self.pools[i](out)

        # Bottleneck
        out = self.bottleneck(out)
        out = self.bottleneck_se(out)

        # Decoder
        for i in range(self.depth):
            out = self.up_convs[i](out)
            skip = skip_connections[-(i+1)]
            if out.shape[-2:] != skip.shape[-2:]:
                out = self._final_interp(out, skip.shape[-2:])
            out = torch.cat([out, skip], dim=1)
            out = self.dec_blocks[i](out)
            if self.dec_se_blocks:
                out = self.dec_se_blocks[i](out)

        raw = self.out_conv(out)
        if raw.shape[-2:] != self.input_shape:
            raw = self._final_interp(raw, self.input_shape)

        # Handle output modes
        if self.output_mode == "mask_sigmoid":
            mask = torch.sigmoid(raw)
            if noisy_mag is not None:
                return {"out_mag": mask * noisy_mag, "mask": mask}
            return {"mask": mask}
        
        elif self.output_mode == "mask_scaled_sigmoid":
            mask = self.mask_max * torch.sigmoid(raw)
            if noisy_mag is not None:
                return {"out_mag": mask * noisy_mag, "mask": mask}
            return {"mask": mask}
        
        elif self.output_mode == "mask_softplus":
            mask = torch.clamp(F.softplus(raw), max=self.softplus_max)
            if noisy_mag is not None:
                return {"out_mag": mask * noisy_mag, "mask": mask}
            return {"mask": mask}
        
        elif self.output_mode == "direct_mag":
            out_mag = F.softplus(raw)
            return {"out_mag": out_mag}
        
        elif self.output_mode == "residual_mag":
            if noisy_mag is not None:
                out_mag = torch.clamp(noisy_mag + raw, min=0)
                return {"out_mag": out_mag}
            return {"residual": raw}
        
        elif self.output_mode == "complex_crm":
            # CRM: complex ratio mask (2 channels: real/imag)
            # out_spec = crm * noisy_spec
            # Assuming raw has 2 channels
            if noisy_spec is not None:
                crm = torch.complex(raw[:, 0], raw[:, 1]).unsqueeze(1) # [B, 1, F, T]
                out_spec = crm * noisy_spec.unsqueeze(1)
                return {"out_spec": out_spec, "crm": crm}
            return {"crm_raw": raw}
        
        elif self.output_mode == "complex_stft":
            # Direct complex output
            out_spec = torch.complex(raw[:, 0], raw[:, 1]).unsqueeze(1)
            return {"out_spec": out_spec}
        
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
