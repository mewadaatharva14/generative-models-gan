"""
Conditional GAN (WGAN-GP) — Discriminator / Critic
=====================================================
Scores images as real or fake, conditioned on class label.

In WGAN-GP the Discriminator is called a CRITIC because:
    - It does NOT output a probability (no Sigmoid)
    - It outputs a raw scalar score (unbounded)
    - Higher score = more real, lower = more fake
    - The Wasserstein distance is estimated from these scores

Why NO Sigmoid?
    BCELoss needs probabilities (0,1) — requires Sigmoid.
    Wasserstein loss works with raw scores — Sigmoid would
    clip gradients and destroy the Wasserstein estimate.

Why NO BatchNorm in Critic?
    WGAN-GP's gradient penalty requires computing gradients
    w.r.t. interpolated inputs. BatchNorm creates dependencies
    between samples in a batch — this corrupts the per-sample
    gradient computation needed for the penalty.
    We use LayerNorm or no normalization instead.

Why SpectralNorm on Conv layers?
    Spectral normalization constrains the Lipschitz constant
    of each layer — a requirement for stable Wasserstein estimation.
    It divides weights by their largest singular value,
    preventing any single layer from having extreme responses.

Label conditioning in Critic:
    Same nn.Embedding approach as Generator.
    Label embedding is projected to (1, 32, 32) and concatenated
    as an extra channel: real input (3, 32, 32) + label (1, 32, 32)
    → critic sees (4, 32, 32). This forces the critic to evaluate
    whether the IMAGE matches the LABEL, not just whether
    it looks realistic in isolation.
"""

import torch
import torch.nn as nn


class ConditionalDiscriminator(nn.Module):
    """
    Convolutional Critic for Conditional WGAN-GP on CIFAR-10.

    Takes an image and a class label, outputs a raw scalar score.
    Higher score = critic thinks image is more real.

    Parameters
    ----------
    image_channels : int — input image channels (3 for RGB)
    num_classes    : int — number of classes (10 for CIFAR-10)
    embedding_dim  : int — label embedding size
    base_channels  : int — base feature map size

    Architecture detail:
        Input  : img (B, 3, 32, 32) + label_map (B, 1, 32, 32)
        Concat : (B, 4, 32, 32)
        Conv   : (B, 64,  16, 16) ← downsample ×2
        Conv   : (B, 128, 8,  8)  ← downsample ×2
        Conv   : (B, 256, 4,  4)  ← downsample ×2
        Flatten: (B, 256*4*4)
        Linear : (B, 1)           ← raw score, no Sigmoid
    """

    def __init__(
        self,
        image_channels: int = 3,
        num_classes:    int = 10,
        embedding_dim:  int = 100,
        base_channels:  int = 64,
    ) -> None:
        super().__init__()

        # ── Label Embedding ───────────────────────────────────────────
        # Maps label integer → dense vector → reshape to spatial map
        # The spatial map is concatenated as an extra image channel
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.label_projection = nn.Linear(embedding_dim, 1 * 32 * 32)

        # ── Convolutional Downsampling Blocks ─────────────────────────
        # Input: (B, 4, 32, 32) — 3 image channels + 1 label channel
        # SpectralNorm on each Conv layer for Lipschitz constraint
        self.conv_blocks = nn.Sequential(

            # Block 1: (B, 4, 32, 32) → (B, 64, 16, 16)
            nn.utils.spectral_norm(
                nn.Conv2d(
                    image_channels + 1, base_channels,
                    kernel_size=4, stride=2, padding=1, bias=False
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (B, 64, 16, 16) → (B, 128, 8, 8)
            nn.utils.spectral_norm(
                nn.Conv2d(
                    base_channels, base_channels * 2,
                    kernel_size=4, stride=2, padding=1, bias=False
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (B, 128, 8, 8) → (B, 256, 4, 4)
            nn.utils.spectral_norm(
                nn.Conv2d(
                    base_channels * 2, base_channels * 4,
                    kernel_size=4, stride=2, padding=1, bias=False
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Output layer ──────────────────────────────────────────────
        # Flatten (B, 256, 4, 4) → (B, 4096) → (B, 1)
        # No Sigmoid — raw Wasserstein score
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(
                nn.Linear(base_channels * 4 * 4 * 4, 1)
            ),
        )

    def forward(
        self,
        img:    torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a batch of images given their class labels.

        Shape trace:
            img            : (B, 3, 32, 32)
            labels         : (B,)
            label_embed    : (B, 100)
            label_proj     : (B, 1024) → (B, 1, 32, 32)
            concat         : (B, 4, 32, 32)
            Conv block 1   : (B, 64,  16, 16)
            Conv block 2   : (B, 128, 8,  8)
            Conv block 3   : (B, 256, 4,  4)
            flatten+linear : (B, 1)  ← raw score

        Parameters
        ----------
        img    : torch.Tensor, shape (B, 3, 32, 32)
        labels : torch.Tensor, shape (B,) — integer class labels

        Returns
        -------
        score : torch.Tensor, shape (B, 1) — raw critic score
        """
        # embed label and reshape to spatial map
        label_embed = self.label_embedding(labels)             # (B, 100)
        label_map   = self.label_projection(label_embed)       # (B, 1024)
        label_map   = label_map.view(
            label_map.size(0), 1, 32, 32
        )                                                       # (B, 1, 32, 32)

        # concatenate image + label map as extra channel
        x = torch.cat([img, label_map], dim=1)                 # (B, 4, 32, 32)

        x = self.conv_blocks(x)                                # (B, 256, 4, 4)
        return self.output(x)                                  # (B, 1)

    def init_weights(self) -> None:
        """
        Apply normal weight initialization to Conv and Linear layers.

        SpectralNorm wraps the weight — we initialize the underlying
        weight_orig attribute which SpectralNorm normalizes at runtime.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)