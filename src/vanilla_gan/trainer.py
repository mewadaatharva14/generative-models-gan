"""
Vanilla GAN — Trainer
======================
Training loop for Vanilla GAN on MNIST.

Per batch:
    1. Train D on real images (label=0.9) + fake images (label=0)
    2. Train G to fool D (label=1 for fake images)

Key decisions:
    fake.detach() in D step   — stops gradients flowing into G
    label smoothing 0.9       — prevents D from overconfidence
    fixed_noise               — same z every epoch for comparable grids
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.vanilla_gan.generator     import Generator
from src.vanilla_gan.discriminator import Discriminator


class VanillaGANTrainer:
    """
    Full training pipeline for Vanilla GAN on MNIST.

    Parameters
    ----------
    config : dict — parsed YAML config

    Example
    -------
    >>> trainer = VanillaGANTrainer(config)
    >>> trainer.train()
    """

    def __init__(self, config: dict) -> None:
        self.config          = config
        self.latent_dim      = config["model"]["latent_dim"]
        self.hidden_dims     = config["model"]["hidden_dims"]
        self.image_dim       = config["model"]["image_dim"]
        self.batch_size      = config["data"]["batch_size"]
        self.num_workers     = config["data"]["num_workers"]
        self.epochs          = config["training"]["epochs"]
        self.lr              = config["training"]["lr"]
        self.betas           = tuple(config["training"]["betas"])
        self.label_smoothing = config["training"]["label_smoothing"]
        self.log_interval    = config["training"]["log_interval"]
        self.save_interval   = config["training"]["save_interval"]
        self.seed            = config["reproducibility"]["random_seed"]
        self.checkpoint_dir  = config["paths"]["checkpoint_dir"]
        self.samples_dir     = config["paths"]["samples_dir"]
        self.assets_dir      = config["paths"]["assets_dir"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)

        for path in [self.checkpoint_dir, self.samples_dir, self.assets_dir]:
            os.makedirs(path, exist_ok=True)

        # models
        self.G = Generator(
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            image_dim=self.image_dim,
        ).to(self.device)

        self.D = Discriminator(image_dim=self.image_dim).to(self.device)

        self.G.init_weights()
        self.D.init_weights()

        self.criterion = nn.BCELoss()

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=self.betas)
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=self.betas)

        # fixed noise — same every epoch so grids are comparable over time
        self.fixed_noise = torch.randn(64, self.latent_dim).to(self.device)

        self.d_losses: list[float] = []
        self.g_losses: list[float] = []

    # ------------------------------------------------------------------

    def _get_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),   # [0,1] → [-1,1]
        ])
        dataset = MNIST(root="data", train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size,
                         shuffle=True, num_workers=self.num_workers)

    def _save_image_grid(self, epoch: int) -> None:
        self.G.eval()
        with torch.no_grad():
            fake = self.G(self.fixed_noise)         # (64, 1, 28, 28)
        path = os.path.join(self.samples_dir, f"epoch_{epoch:03d}.png")
        torchvision.utils.save_image(fake, path, nrow=8, normalize=True,
                                     value_range=(-1, 1))
        self.G.train()

    def _save_checkpoint(self, epoch: int, g_loss: float) -> None:
        ckpt = {
            "epoch":        epoch,
            "g_state_dict": self.G.state_dict(),
            "d_state_dict": self.D.state_dict(),
            "opt_g":        self.opt_G.state_dict(),
            "opt_d":        self.opt_D.state_dict(),
            "config":       self.config,
        }
        torch.save(ckpt, os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth"
        ))
        if not self.g_losses or g_loss <= min(self.g_losses):
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "best_generator.pth"))
            print(f"  ✓ Best generator saved (G loss: {g_loss:.4f})")

    def _save_loss_plot(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.d_losses, color="#E74C3C", linewidth=1.8, label="D Loss")
        ax.plot(self.g_losses, color="#1A6BC4", linewidth=1.8, label="G Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Vanilla GAN — Training Loss Curves (MNIST)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.assets_dir, "training_curves.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"Training curves saved → {path}")

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full Vanilla GAN training loop."""

        print("\n" + "="*55)
        print("  Vanilla GAN — MNIST")
        print("="*55)
        print(f"  Device        : {self.device}")
        print(f"  Generator     : {self.G.count_parameters():,} params")
        print(f"  Discriminator : {self.D.count_parameters():,} params")
        print(f"  Epochs        : {self.epochs}")
        print("="*55 + "\n")

        loader = self._get_dataloader()

        for epoch in range(1, self.epochs + 1):
            epoch_d, epoch_g, n_batches = 0.0, 0.0, 0

            pbar = tqdm(loader, desc=f"Epoch {epoch:>3}/{self.epochs}", ncols=85)

            for real_imgs, _ in pbar:
                real_imgs = real_imgs.to(self.device)
                n = real_imgs.size(0)

                real_labels = torch.full((n, 1), self.label_smoothing).to(self.device)
                fake_labels = torch.zeros(n, 1).to(self.device)

                # ── Train Discriminator ───────────────────────────────
                self.opt_D.zero_grad()

                d_real  = self.D(real_imgs)
                loss_real = self.criterion(d_real, real_labels)

                z    = torch.randn(n, self.latent_dim).to(self.device)
                fake = self.G(z)
                d_fake    = self.D(fake.detach())
                loss_fake = self.criterion(d_fake, fake_labels)

                d_loss = loss_real + loss_fake
                d_loss.backward()
                self.opt_D.step()

                # ── Train Generator ───────────────────────────────────
                self.opt_G.zero_grad()

                d_fake_for_g = self.D(fake)
                g_loss = self.criterion(
                    d_fake_for_g,
                    torch.ones(n, 1).to(self.device)
                )
                g_loss.backward()
                self.opt_G.step()

                epoch_d  += d_loss.item()
                epoch_g  += g_loss.item()
                n_batches += 1

                pbar.set_postfix({
                    "D": f"{d_loss.item():.3f}",
                    "G": f"{g_loss.item():.3f}",
                })

            avg_d = epoch_d / n_batches
            avg_g = epoch_g / n_batches
            self.d_losses.append(avg_d)
            self.g_losses.append(avg_g)

            print(f"  Epoch {epoch:>3}/{self.epochs} | "
                  f"D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")

            if epoch % self.save_interval == 0:
                self._save_image_grid(epoch)

            self._save_checkpoint(epoch, avg_g)

        # post-training
        self._save_loss_plot()

        self.G.eval()
        with torch.no_grad():
            final = self.G(self.fixed_noise)
        torchvision.utils.save_image(
            final,
            os.path.join(self.assets_dir, "final_generated_grid.png"),
            nrow=8, normalize=True, value_range=(-1, 1)
        )
        print(f"\nFinal grid → {self.assets_dir}/final_generated_grid.png")
        print(f"Best checkpoint → {self.checkpoint_dir}/best_generator.pth\n")

    def get_history(self) -> dict:
        return {"d_losses": self.d_losses, "g_losses": self.g_losses}