# unlearnable_unet.py

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from typing import Optional
from dataclasses import dataclass

@dataclass
class UNetOutput:
    sample: torch.FloatTensor

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class UnlearnableUnetModel(nn.Module):
    """
    Simplified unconditional UNet:
      - Accepts latent tensor of shape (B, C, H, W)
      - Ignores timestep & encoder_hidden_states
      - Returns UNetOutput(sample=prediction)
    """
    def __init__(
        self,
        in_channels: int,
        base_features: Optional[list[int]] = None
    ):
        super().__init__()
        if base_features is None:
            base_features = [64, 128, 256, 512]

        # down / encoder
        self.downs = nn.ModuleList()
        ch = in_channels
        for out_ch in base_features:
            self.downs.append(DoubleConv(ch, out_ch))
            ch = out_ch
        self.pool = nn.MaxPool2d(2, 2)

        # bottleneck
        self.bottleneck = DoubleConv(ch, ch * 2)

        # up / decoder
        self.ups = nn.ModuleList()
        for out_ch in reversed(base_features):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(ch * 2, out_ch))
            ch = out_ch

        # final conv: map back to latent channels
        self.final_conv = nn.Conv2d(base_features[0], in_channels, kernel_size=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None
    ) -> UNetOutput:
        # Encoder
        skips = []
        x = sample
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for idx in range(0, len(self.ups), 2):
            deconv = self.ups[idx]
            conv   = self.ups[idx + 1]
            skip   = skips[-(idx//2 + 1)]

            x = deconv(x)
            # if shape mismatch due to odd dims
            if x.shape[2:] != skip.shape[2:]:
                x = TF.resize(x, size=skip.shape[2:])
            x = conv(torch.cat([skip, x], dim=1))

        # Final prediction
        out = self.final_conv(x)
        return UNetOutput(sample=out)