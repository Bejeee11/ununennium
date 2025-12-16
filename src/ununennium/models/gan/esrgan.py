"""ESRGAN: Enhanced Super-Resolution Generative Adversarial Network.

This module implements ESRGAN for satellite imagery super-resolution.

References:
    Wang, X., et al. (2018). ESRGAN: Enhanced Super-Resolution Generative
    Adversarial Networks. ECCV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for ESRGAN.
    
    Parameters
    ----------
    num_features : int
        Number of input/output features.
    growth_rate : int
        Growth rate for dense connections.
    """
    
    def __init__(self, num_features: int = 64, growth_rate: int = 32) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_features, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features + 2 * growth_rate, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features + 3 * growth_rate, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_features + 4 * growth_rate, num_features, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.beta = 0.2  # Residual scaling
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * self.beta + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block.
    
    Parameters
    ----------
    num_features : int
        Number of features.
    growth_rate : int
        Growth rate for RDB.
    """
    
    def __init__(self, num_features: int = 64, growth_rate: int = 32) -> None:
        super().__init__()
        
        self.rdb1 = ResidualDenseBlock(num_features, growth_rate)
        self.rdb2 = ResidualDenseBlock(num_features, growth_rate)
        self.rdb3 = ResidualDenseBlock(num_features, growth_rate)
        self.beta = 0.2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.beta + x


class ESRGANGenerator(nn.Module):
    """ESRGAN Generator network.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_features : int
        Number of intermediate features.
    num_rrdb : int
        Number of RRDB blocks.
    scale_factor : int
        Upscaling factor (2, 4, or 8).
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_rrdb: int = 23,
        scale_factor: int = 4,
    ) -> None:
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # First convolution
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # RRDB trunk
        self.trunk = nn.Sequential(*[
            RRDB(num_features) for _ in range(num_rrdb)
        ])
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        # Upsampling
        upsample_layers = []
        if scale_factor in [2, 4, 8]:
            n_upsample = {2: 1, 4: 2, 8: 3}[scale_factor]
            for _ in range(n_upsample):
                upsample_layers += [
                    nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
        self.upsampler = nn.Sequential(*upsample_layers)
        
        # High-resolution convolutions
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(feat))
        feat = feat + trunk
        
        feat = self.upsampler(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class ESRGANDiscriminator(nn.Module):
    """ESRGAN VGG-style Discriminator.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    num_features : int
        Base number of features.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
    ) -> None:
        super().__init__()
        
        def conv_block(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv_block(num_features, num_features, 2),
            conv_block(num_features, num_features * 2),
            conv_block(num_features * 2, num_features * 2, 2),
            conv_block(num_features * 2, num_features * 4),
            conv_block(num_features * 4, num_features * 4, 2),
            conv_block(num_features * 4, num_features * 8),
            conv_block(num_features * 8, num_features * 8, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = self.classifier(feat)
        return out


class ESRGAN(nn.Module):
    """ESRGAN for super-resolution.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    scale_factor : int
        Upscaling factor (2, 4, or 8).
    num_features : int
        Number of features in generator.
    num_rrdb : int
        Number of RRDB blocks.
        
    Examples
    --------
    >>> model = ESRGAN(in_channels=12, scale_factor=4)
    >>> lr_image = torch.randn(1, 12, 64, 64)
    >>> hr_image = model.generator(lr_image)
    >>> hr_image.shape
    torch.Size([1, 12, 256, 256])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        scale_factor: int = 4,
        num_features: int = 64,
        num_rrdb: int = 23,
    ) -> None:
        super().__init__()
        
        self.generator = ESRGANGenerator(
            in_channels=in_channels,
            out_channels=in_channels,
            num_features=num_features,
            num_rrdb=num_rrdb,
            scale_factor=scale_factor,
        )
        
        self.discriminator = ESRGANDiscriminator(
            in_channels=in_channels,
            num_features=num_features,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate super-resolved image."""
        return self.generator(x)
