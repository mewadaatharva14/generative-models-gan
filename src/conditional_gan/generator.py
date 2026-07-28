"""
Conditional GAN — Generator
=============================
Generates CIFAR-10 images conditioned on a class label.

Architecture:
    z (100,) + label → Embedding → Concat → Linear →
    Reshape (256,4,4) → ConvTranspose2d ×3 → Tanh → (3,32,32)
"""

import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    """
    Convolutional Generator for Conditional WGAN-GP on CIFAR-10.

    Parameters
    ----------
    latent_dim     : int — noise vector size (default 100)
    embedding_dim  : int — label embedding size (default 100)
    num_classes    : int — number of classes (default 10)
    base_channels  : int — base feature map channels (default 256)
    image_channels : int — output channels, 3 for RGB
    """

    def __init__(
        self,
        latent_dim:     int = 100,
        embedding_dim:  int = 100,
        num_classes:    int = 10,
        base_channels:  int = 256,
        image_channels: int = 3,
    ) -> None:
        super().__init__()

        self.latent_dim    = latent_dim
        self.base_channels = base_channels

        # label → dense vector, trained jointly with Generator
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # [z | label_embed] (200,) → (256*4*4,) → reshape to (256,4,4)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, base_channels * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # upsample: 4→8→16→32
        self.conv_blocks = nn.Sequential(
            # (B, 256, 4, 4) → (B, 128, 8, 8)
            nn.ConvTranspose2d(base_channels, base_channels // 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 128, 8, 8) → (B, 64, 16, 16)
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, 16, 16) → (B, 3, 32, 32)
            nn.ConvTranspose2d(base_channels // 4, image_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z      : (B, latent_dim)
        labels : (B,) — integer class labels [0,9]

        Returns
        -------
        img : (B, 3, 32, 32) — pixel values in [-1, 1]
        """
        label_embed = self.label_embedding(labels)          # (B, 100)
        x = torch.cat([z, label_embed], dim=1)             # (B, 200)
        x = self.fc(x)                                     # (B, 4096)
        x = x.view(x.size(0), self.base_channels, 4, 4)   # (B, 256, 4, 4)
        return self.conv_blocks(x)                         # (B, 3, 32, 32)

    def init_weights(self) -> None:
        """Normal init: mean=0, std=0.02."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)