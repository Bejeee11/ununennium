"""Siamese network for change detection.

This module implements Siamese architectures for bitemporal change detection.

References:
    Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully Convolutional
    Siamese Networks for Change Detection. ICIP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class SiameseEncoder(nn.Module):
    """Shared encoder for Siamese change detection.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    base_channels : int
        Base number of channels.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        
        self.enc1 = self._make_layer(in_channels, base_channels)
        self.enc2 = self._make_layer(base_channels, base_channels * 2)
        self.enc3 = self._make_layer(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_layer(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def _make_layer(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        
        x1 = self.enc1(x)
        features.append(x1)
        x = self.pool(x1)
        
        x2 = self.enc2(x)
        features.append(x2)
        x = self.pool(x2)
        
        x3 = self.enc3(x)
        features.append(x3)
        x = self.pool(x3)
        
        x4 = self.enc4(x)
        features.append(x4)
        
        return features


class SiameseDecoder(nn.Module):
    """Decoder for Siamese change detection.
    
    Parameters
    ----------
    base_channels : int
        Base number of channels (should match encoder).
    num_classes : int
        Number of output classes.
    fusion : str
        Fusion strategy: "diff", "concat", "dist".
    """
    
    def __init__(
        self,
        base_channels: int = 64,
        num_classes: int = 2,
        fusion: Literal["diff", "concat", "dist"] = "diff",
    ) -> None:
        super().__init__()
        
        self.fusion = fusion
        
        # Fusion multiplier for channel computation
        mult = 2 if fusion == "concat" else 1
        
        self.dec4 = self._make_layer(base_channels * 8 * mult, base_channels * 4)
        self.dec3 = self._make_layer(base_channels * 4 * (mult + 1), base_channels * 2)
        self.dec2 = self._make_layer(base_channels * 2 * (mult + 1), base_channels)
        self.dec1 = self._make_layer(base_channels * (mult + 1), base_channels)
        
        self.out_conv = nn.Conv2d(base_channels, num_classes, 1)
        
    def _make_layer(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        
    def _fuse(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        if self.fusion == "diff":
            return torch.abs(f2 - f1)
        elif self.fusion == "concat":
            return torch.cat([f1, f2], dim=1)
        elif self.fusion == "dist":
            return torch.sqrt((f2 - f1) ** 2 + 1e-8)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
        
    def forward(
        self, 
        features_pre: list[torch.Tensor], 
        features_post: list[torch.Tensor],
    ) -> torch.Tensor:
        # Fuse deepest features
        f4 = self._fuse(features_pre[3], features_post[3])
        x = self.dec4(f4)
        
        # Upsample and fuse
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        f3 = self._fuse(features_pre[2], features_post[2])
        x = torch.cat([x, f3], dim=1)
        x = self.dec3(x)
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        f2 = self._fuse(features_pre[1], features_post[1])
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        f1 = self._fuse(features_pre[0], features_post[0])
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)
        
        return self.out_conv(x)


class SiameseChangeDetection(nn.Module):
    """Siamese network for change detection.
    
    A Siamese architecture that processes two temporal images through
    a shared encoder and fuses features for change map prediction.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels per temporal image.
    num_classes : int
        Number of output classes (2 for binary change detection).
    base_channels : int
        Base number of channels in encoder/decoder.
    fusion : str
        Fusion strategy: "diff" (difference), "concat" (concatenation),
        or "dist" (euclidean distance).
        
    Examples
    --------
    >>> model = SiameseChangeDetection(in_channels=12, num_classes=2)
    >>> pre = torch.randn(1, 12, 256, 256)
    >>> post = torch.randn(1, 12, 256, 256)
    >>> change_map = model(pre, post)
    >>> change_map.shape
    torch.Size([1, 2, 256, 256])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 64,
        fusion: Literal["diff", "concat", "dist"] = "diff",
    ) -> None:
        super().__init__()
        
        self.encoder = SiameseEncoder(in_channels, base_channels)
        self.decoder = SiameseDecoder(base_channels, num_classes, fusion)
        
    def forward(
        self, 
        x_pre: torch.Tensor, 
        x_post: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x_pre : torch.Tensor
            Pre-change image, shape (B, C, H, W).
        x_post : torch.Tensor
            Post-change image, shape (B, C, H, W).
            
        Returns
        -------
        torch.Tensor
            Change map logits, shape (B, num_classes, H, W).
        """
        features_pre = self.encoder(x_pre)
        features_post = self.encoder(x_post)
        
        return self.decoder(features_pre, features_post)


# Alias for consistency with API documentation
SiameseUNet = SiameseChangeDetection
