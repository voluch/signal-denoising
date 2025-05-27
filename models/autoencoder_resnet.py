import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        # Encoder
        self.enc1 = ResidualBlock(1, 32, downsample=False)
        self.enc2 = ResidualBlock(32, 64, downsample=True)
        self.enc3 = ResidualBlock(64, 128, downsample=True)

        # Decoder (transpose conv upsample)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d1 = self.up1(e3)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = self.dec2(d2)
        out = self.final(d2)
        out = self.sigmoid(out)
        # If needed, resize to input_shape (handles rounding)
        if out.shape[-2:] != self.input_shape:
            out = nn.functional.interpolate(out, size=self.input_shape, mode='bilinear', align_corners=False)
        return out
