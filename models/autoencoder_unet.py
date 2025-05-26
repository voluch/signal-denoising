import torch
import torch.nn as nn
import torch.nn.functional as F


# Автоенкодер для спектрограм
class UnetAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),   # no downsampling in freq
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(1, 2), padding=1),  # only downsample in time
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.input_shape = input_shape

    @staticmethod
    def crop_to_match(enc_feat, target_feat):
        _, _, h, w = enc_feat.shape
        _, _, th, tw = target_feat.shape
        dh = h - th
        dw = w - tw
        enc_feat = enc_feat[..., dh // 2:h - (dh - dh // 2), dw // 2:w - (dw - dw // 2)]
        return enc_feat

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        d1 = self.dec1(e2)
        if d1.shape[-2:] != e1.shape[-2:]:
            e1 = self.crop_to_match(e1, d1)
        d1 = torch.cat([d1, e1], dim=1)

        out = self.final_conv(d1)
        out = self.sigmoid(out)

        if out.shape[-2:] != self.input_shape:
            out = F.interpolate(out, size=self.input_shape, mode='bilinear', align_corners=False)
        return out

