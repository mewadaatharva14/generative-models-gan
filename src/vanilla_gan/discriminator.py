"""
Vanilla GAN — Discriminator
=============================
Classifies images as real (1) or fake (0).

Architecture:
    x (784,) → Linear → LeakyReLU → Dropout → ... → Linear → Sigmoid → (1,)

Why Sigmoid at output?
    Discriminator outputs a probability in (0, 1).
    BCELoss expects probabilities — Sigmoid provides them.
    In WGAN-GP we remove Sigmoid (critic outputs raw scores),
    but for Vanilla GAN with BCELoss, Sigmoid is required.

Why Dropout in Discriminator but not Generator?
    Dropout prevents the Discriminator from becoming too powerful
    too quickly. If D becomes perfect early, it gives G zero gradient
    — G cannot learn. Dropout keeps D slightly uncertain,
    giving G a learning signal throughout training.

Why NO BatchNorm in Discriminator?
    BatchNorm normalizes across the batch, mixing information
    between real and fake samples. This corrupts the Discriminator's
    ability to distinguish their distributions cleanly.
    LeakyReLU alone provides sufficient non-linearity.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Fully connected Discriminator for Vanilla GAN on MNIST.

    Takes a flat 784-dim image vector and outputs a single
    probability score — real (→1) or fake (→0).

    Parameters
    ----------
    image_dim : int — input size = 28*28 = 784
    dropout   : float — dropout probability (default 0.3)
    """

    def __init__(
        self,
        image_dim: int   = 784,
        dropout:   float = 0.3,
    ) -> None:
        super().__init__()

        self.image_dim = image_dim

        self.model = nn.Sequential(
            # ── Block 1 ───────────────────────────────────────────────
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),              # keep D from dominating too early

            # ── Block 2 ───────────────────────────────────────────────
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            # ── Output ────────────────────────────────────────────────
            nn.Linear(256, 1),
            nn.Sigmoid()                      # output ∈ (0,1) — probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of images as real or fake.

        Shape trace:
            x (images) : (batch, 1, 28, 28)
            → view     : (batch, 784)       ← flatten spatial dims
            → Linear   : (batch, 512)
            → LReLU+DO : (batch, 512)
            → Linear   : (batch, 256)
            → LReLU+DO : (batch, 256)
            → Linear   : (batch, 1)
            → Sigmoid  : (batch, 1)  values in (0, 1)

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, 1, 28, 28)
            or (batch_size, 784) — both handled

        Returns
        -------
        score : torch.Tensor, shape (batch_size, 1)
                probability of being a real image
        """
        # flatten if image tensor passed — handles both shapes
        if x.dim() == 4:
            x = x.view(x.size(0), -1)         # (batch, 1, 28, 28) → (batch, 784)
        return self.model(x)                   # (batch, 1)

    def init_weights(self) -> None:
        """
        Apply normal weight initialization to Linear layers.

        mean=0, std=0.02 — standard for GAN initialization.
        No BatchNorm in Discriminator so only Linear layers initialized.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
