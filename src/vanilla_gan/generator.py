"""
Vanilla GAN — Generator
========================
Maps a random noise vector z → fake MNIST image (28x28 grayscale).

Architecture : z (100,) → Linear → BN → LeakyReLU → ... → Tanh → (784,)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Fully connected Generator for Vanilla GAN on MNIST.

    Parameters
    ----------
    latent_dim  : int       — size of input noise vector (default 100)
    hidden_dims : list[int] — hidden layer sizes (default [256, 512, 1024])
    image_dim   : int       — output size 28*28=784
    """

    def __init__(
        self,
        latent_dim:  int       = 100,
        hidden_dims: list[int] = None,
        image_dim:   int       = 784,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]

        self.latent_dim = latent_dim

        # build layers from hidden_dims — architecture controlled by config
        layers    = []
        in_dim    = latent_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_dim = h_dim

        layers += [nn.Linear(in_dim, image_dim), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        img : (B, 1, 28, 28) — pixel values in [-1, 1]
        """
        out = self.model(z)                             # (B, 784)
        return out.view(out.size(0), 1, 28, 28)         # (B, 1, 28, 28)

    def init_weights(self) -> None:
        """Normal init: mean=0, std=0.02 — standard GAN practice."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)