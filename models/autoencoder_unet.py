import torch
import torch.nn as nn
import torch.nn.functional as F


# Автоенкодер для спектрограм
class UnetAutoencoder(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.input_shape = input_shape  # For final upsampling

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, 16, H, W]
        e2 = self.enc2(e1)  # [B, 32, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 64, H/4, W/4]
        e4 = self.enc4(e3)  # [B, 128, H/8, W/8]

        # Decoder with skip connections (concatenate along channels)
        d3 = self.dec3(e4)  # [B, 64, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)  # [B, 128, H/4, W/4]
        d2 = self.dec2(d3)  # [B, 32, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)  # [B, 64, H/2, W/2]
        d1 = self.dec1(d2)  # [B, 16, H, W]
        d1 = torch.cat([d1, e1], dim=1)  # [B, 32, H, W]

        out = self.final_conv(d1)  # [B, 1, H, W]
        out = self.sigmoid(out)

        # Final upsample if necessary
        if out.shape[-2:] != self.input_shape:
            out = F.interpolate(out, size=self.input_shape, mode='bilinear', align_corners=False)
        return out
