"""
WGAN-GP — Gradient Penalty
============================
Enforces the Lipschitz constraint on the critic via gradient penalty.

Why Lipschitz constraint?
    Wasserstein distance requires the critic function to be
    1-Lipschitz: |f(x) - f(y)| ≤ |x - y| for all x, y.
    Without this constraint, the critic can become arbitrarily
    large, making the Wasserstein estimate meaningless.

Original WGAN used weight clipping to enforce this.
WGAN-GP replaces clipping with a gradient penalty because:
    - Weight clipping causes vanishing/exploding gradients
    - Clipping limits model capacity
    - Gradient penalty is softer and more stable

How the penalty works:
    1. Sample interpolated points between real and fake images:
       x_hat = epsilon * real + (1 - epsilon) * fake
       where epsilon ~ Uniform(0, 1) per sample

    2. Compute critic scores on interpolated points:
       scores = critic(x_hat, labels)

    3. Compute gradients of scores w.r.t. x_hat:
       gradients = d(scores) / d(x_hat)

    4. Compute gradient norm per sample:
       grad_norm = ||gradients||_2

    5. Penalty = mean((grad_norm - 1)^2)
       We penalize deviation from norm=1, not norm>1.
       This enforces the 1-Lipschitz constraint exactly.

    6. Total critic loss = Wasserstein loss + lambda * penalty
       lambda=10 is the standard value from the WGAN-GP paper.
"""

import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad


def compute_gradient_penalty(
    critic:     nn.Module,
    real_imgs:  torch.Tensor,
    fake_imgs:  torch.Tensor,
    labels:     torch.Tensor,
    device:     torch.device,
    lambda_gp:  float = 10.0,
) -> torch.Tensor:
    """
    Compute WGAN-GP gradient penalty.

    Samples random interpolations between real and fake images,
    computes critic gradients w.r.t. those interpolations,
    and penalizes deviation of gradient norm from 1.

    Parameters
    ----------
    critic     : nn.Module — the critic/discriminator model
    real_imgs  : torch.Tensor, shape (B, C, H, W) — real images
    fake_imgs  : torch.Tensor, shape (B, C, H, W) — generated images
    labels     : torch.Tensor, shape (B,) — class labels
    device     : torch.device
    lambda_gp  : float — gradient penalty weight (default 10.0)

    Returns
    -------
    penalty : torch.Tensor, scalar — weighted gradient penalty
              to be added to critic loss

    Shape trace:
        epsilon     : (B, 1, 1, 1)  — broadcast over C, H, W
        interpolated: (B, C, H, W)  — convex combo of real + fake
        scores      : (B, 1)        — critic output on interpolated
        gradients   : (B, C, H, W)  — d(scores)/d(interpolated)
        grad_norm   : (B,)          — L2 norm per sample
        penalty     : scalar        — mean((grad_norm - 1)^2) * lambda
    """

    batch_size = real_imgs.size(0)

    # ── Step 1: Sample random interpolation coefficients ─────────────
    # epsilon shape (B, 1, 1, 1) broadcasts across C, H, W automatically
    # Each sample in the batch gets its own epsilon
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)

    # ── Step 2: Compute interpolated images ──────────────────────────
    # x_hat = epsilon * real + (1 - epsilon) * fake
    # requires_grad=True: we need gradients w.r.t. x_hat for the penalty
    interpolated = (
        epsilon * real_imgs + (1 - epsilon) * fake_imgs.detach()
    ).requires_grad_(True)

    # ── Step 3: Critic scores on interpolated images ──────────────────
    critic_scores = critic(interpolated, labels)           # (B, 1)

    # ── Step 4: Compute gradients w.r.t. interpolated images ─────────
    # torch_grad computes d(critic_scores) / d(interpolated)
    #
    # grad_outputs=torch.ones_like(critic_scores):
    #   critic_scores is (B,1) not scalar — grad_outputs provides
    #   the vector to left-multiply in the Jacobian-vector product.
    #   ones_like sums gradients across the batch dimension.
    #
    # create_graph=True:
    #   allows second-order gradients — needed because this penalty
    #   is itself differentiated during critic's backward pass.
    #
    # retain_graph=True:
    #   keeps the computation graph alive for subsequent backward calls.
    gradients = torch_grad(
        outputs      = critic_scores,
        inputs       = interpolated,
        grad_outputs = torch.ones_like(critic_scores),
        create_graph = True,
        retain_graph = True,
    )[0]                                                   # (B, C, H, W)

    # ── Step 5: Compute gradient norm per sample ──────────────────────
    # Flatten spatial dims: (B, C, H, W) → (B, C*H*W)
    # Compute L2 norm across the flattened dim for each sample
    gradients = gradients.view(batch_size, -1)             # (B, C*H*W)
    grad_norm = gradients.norm(2, dim=1)                   # (B,) L2 norm

    # ── Step 6: Penalty = mean((||grad|| - 1)^2) * lambda ────────────
    # We penalize deviation from norm=1 in both directions.
    # (norm - 1)^2 is minimized when norm = 1 exactly.
    penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()

    return penalty