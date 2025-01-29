
import os
os.sys.path.append('../torch-conv-kan')
from kan_convs import FastKANConv2DLayer
import torch
import torch.nn.functional as F

class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = FastKANConv2DLayer(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = FastKANConv2DLayer(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )
        # Input channels = skip_connection_channels + upsampled_channels
        self.conv1 = FastKANConv2DLayer(out_channels * 2, out_channels, 3, padding=1)
        self.conv2 = FastKANConv2DLayer(out_channels, out_channels, 3, padding=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample decoder input

        # Handle spatial dimension mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY > 0 or diffX > 0:
            # Pad x1 if smaller than x2
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        else:
            # Crop x2 if smaller than x1
            x2 = x2[:, :,
                    -diffY // 2 : x2.size(2) - (-diffY // 2),
                    -diffX // 2 : x2.size(3) - (-diffX // 2)]

        x = torch.cat([x2, x1], dim=1)  # Correct channel dimension
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNetKAN(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(in_channels, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)
        self.pool = torch.nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            FastKANConv2DLayer(256, 512, 3, padding=1),
            FastKANConv2DLayer(512, 512, 3, padding=1)
        )

        # Decoder
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 32)

        # Final output
        self.final_conv = torch.nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)  # [B, 32, H, W]
        x = self.pool(s1)
        s2 = self.enc2(x)  # [B, 64, H/2, W/2]
        x = self.pool(s2)
        s3 = self.enc3(x)  # [B, 128, H/4, W/4]
        x = self.pool(s3)
        s4 = self.enc4(x)  # [B, 256, H/8, W/8]
        x = self.pool(s4)

        # Bottleneck
        x = self.bottleneck(x)  # [B, 512, H/16, W/16]

        # Decoder
        x = self.dec4(x, s4)  # [B, 256, H/8, W/8]
        x = self.dec3(x, s3)  # [B, 128, H/4, W/4]
        x = self.dec2(x, s2)  # [B, 64, H/2, W/2]
        x = self.dec1(x, s1)  # [B, 32, H, W]

        return self.final_conv(x)  # [B, out_channels, H, W]