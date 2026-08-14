import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Double Convolution block with Batch Normalization and ReLU."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ShallowUNet(nn.Module):
    """Custom Shallow U-Net optimized for 8x8 micro-grids."""
    def __init__(self, in_channels=5, num_classes=4):
        super(ShallowUNet, self).__init__()
        # Encoder (Down)
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Down to 4x4
        self.enc2 = DoubleConv(32, 64)
        
        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)
        
        # Decoder (Up)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # Up to 8x8
        self.dec1 = DoubleConv(96, 32)  # Concat: 64 + 32 = 96 channels
        
        # Final Classifier
        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        
        # Bottleneck
        b = self.bottleneck(x2)
        
        # Decoder
        u1 = self.up1(b)
        c1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(c1)
        
        return self.final_conv(d1)
