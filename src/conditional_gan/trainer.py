"""
Conditional WGAN-GP — Trainer
===============================
Training loop for Conditional WGAN-GP on CIFAR-10.

Per batch:
    1. Train Critic × critic_iterations
       C loss = C(fake) - C(real) + lambda * GP
    2. Train Generator × 1
       G loss = -C(G(z, label), label)
"""

import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.conditional_gan.generator        import ConditionalGenerator
from src.conditional_gan.discriminator    import ConditionalDiscriminator
from src.conditional_gan.gradient_penalty import compute_gradient_penalty

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird",  "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck",
]


class ConditionalGANTrainer:
    """
    Full training pipeline for Conditional WGAN-GP on CIFAR-10.

    Parameters
    ----------
    config : dict — parsed YAML config

    Example
    -------
    >>> trainer = ConditionalGANTrainer(config)
    >>> trainer.train()
    """

    def __init__(self, config: dict) -> None:
        self.config            = config
        self.latent_dim        = config["model"]["latent_dim"]
        self.embedding_dim     = config["model"]["embedding_dim"]
        self.base_channels     = config["model"]["base_channels"]
        self.num_classes       = config["data"]["num_classes"]
        self.image_channels    = config["data"]["image_channels"]
        self.batch_size        = config["data"]["batch_size"]
        self.num_workers       = config["data"]["num_workers"]
        self.epochs            = config["training"]["epochs"]
        self.lr                = config["training"]["lr"]
        self.betas             = tuple(config["training"]["betas"])
        self.lambda_gp         = config["training"]["lambda_gp"]
        self.critic_iterations = config["training"]["critic_iterations"]
        self.log_interval      = config["training"]["log_interval"]
        self.save_interval     = config["training"]["save_interval"]
        self.seed              = config["reproducibility"]["random_seed"]
        self.checkpoint_dir    = config["paths"]["checkpoint_dir"]
        self.samples_dir       = config["paths"]["samples_dir"]
        self.assets_dir        = config["paths"]["assets_dir"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)

        for path in [self.checkpoint_dir, self.samples_dir, self.assets_dir]:
            os.makedirs(path, exist_ok=True)

        # models
        self.G = ConditionalGenerator(
            latent_dim=self.latent_dim,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
            image_channels=self.image_channels,
        ).to(self.device)

        self.C = ConditionalDiscriminator(
            image_channels=self.image_channels,
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
        ).to(self.device)

        self.G.init_weights()
        self.C.init_weights()

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=self.betas)
        self.opt_C = optim.Adam(self.C.parameters(), lr=self.lr, betas=self.betas)

        # fixed noise — 8 samples per class, same every epoch
        noise_list, label_list = [], []
        for cls in range(self.num_classes):
            noise_list.append(torch.randn(8, self.latent_dim))
            label_list.append(torch.full((8,), cls, dtype=torch.long))

        self.fixed_noise  = torch.cat(noise_list).to(self.device)   # (80, 100)
        self.fixed_labels = torch.cat(label_list).to(self.device)   # (80,)

        self.c_losses:    list[float] = []
        self.g_losses:    list[float] = []
        self.w_distances: list[float] = []

    # ------------------------------------------------------------------

    def _get_dataloader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = CIFAR10(root="data", train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size,
                         shuffle=True, num_workers=self.num_workers,
                         drop_last=True)

    def _save_image_grid(self, epoch: int) -> None:
        self.G.eval()
        with torch.no_grad():
            fake = self.G(self.fixed_noise, self.fixed_labels)  # (80, 3, 32, 32)

        # per-epoch sample grid (gitignored)
        path = os.path.join(self.samples_dir, f"epoch_{epoch:03d}.png")
        torchvision.utils.save_image(fake, path, nrow=8, normalize=True,
                                     value_range=(-1, 1))
        self.G.train()

    def _save_labeled_grid(self, imgs: torch.Tensor) -> None:
        """Save final grid with CIFAR-10 class name labels per row."""
        imgs = (imgs.cpu().clamp(-1, 1) + 1) / 2.0             # [-1,1] → [0,1]

        fig, axes = plt.subplots(self.num_classes, 8,
                                 figsize=(16, self.num_classes * 2))

        for cls in range(self.num_classes):
            for col in range(8):
                img = imgs[cls * 8 + col].permute(1, 2, 0).numpy()
                axes[cls, col].imshow(img)
                axes[cls, col].axis("off")
            axes[cls, 0].set_ylabel(CIFAR10_CLASSES[cls], fontsize=11,
                                    rotation=0, labelpad=55, va="center")

        plt.suptitle("Conditional WGAN-GP — Generated Images per Class", fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.assets_dir, "final_labeled_grid.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Labeled grid saved → {path}")

    def _save_checkpoint(self, epoch: int, g_loss: float) -> None:
        ckpt = {
            "epoch":        epoch,
            "g_state_dict": self.G.state_dict(),
            "c_state_dict": self.C.state_dict(),
            "opt_g":        self.opt_G.state_dict(),
            "opt_c":        self.opt_C.state_dict(),
            "config":       self.config,
        }
        torch.save(ckpt, os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth"
        ))
        if not self.g_losses or g_loss <= min(self.g_losses):
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "best_generator.pth"))
            print(f"  ✓ Best generator saved (G loss: {g_loss:.4f})")

    def _save_loss_plot(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, data, color, title in zip(
            axes,
            [self.c_losses, self.g_losses, self.w_distances],
            ["#E74C3C",     "#1A6BC4",     "#00C28B"],
            ["Critic Loss", "Generator Loss", "Wasserstein Distance"],
        ):
            ax.plot(data, color=color, linewidth=1.8)
            ax.set_xlabel("Epoch"); ax.set_title(title)
            ax.grid(alpha=0.3)

        plt.suptitle("Conditional WGAN-GP — Training Curves (CIFAR-10)", fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.assets_dir, "training_curves.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"Training curves saved → {path}")

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full Conditional WGAN-GP training loop."""

        print("\n" + "="*55)
        print("  Conditional WGAN-GP — CIFAR-10")
        print("="*55)
        print(f"  Device             : {self.device}")
        print(f"  Generator          : {self.G.count_parameters():,} params")
        print(f"  Critic             : {self.C.count_parameters():,} params")
        print(f"  Epochs             : {self.epochs}")
        print(f"  Critic iterations  : {self.critic_iterations}")
        print(f"  Lambda GP          : {self.lambda_gp}")
        print("="*55 + "\n")

        loader = self._get_dataloader()

        for epoch in range(1, self.epochs + 1):
            epoch_c, epoch_g, epoch_w = 0.0, 0.0, 0.0
            n_batches = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch:>3}/{self.epochs}", ncols=90)

            for real_imgs, labels in pbar:
                real_imgs = real_imgs.to(self.device)
                labels    = labels.to(self.device)
                n         = real_imgs.size(0)

                # ── Train Critic × critic_iterations ──────────────────
                for _ in range(self.critic_iterations):
                    self.opt_C.zero_grad()

                    z    = torch.randn(n, self.latent_dim).to(self.device)
                    fake = self.G(z, labels)

                    c_real = self.C(real_imgs, labels)
                    c_fake = self.C(fake.detach(), labels)

                    gp = compute_gradient_penalty(
                        self.C, real_imgs, fake, labels,
                        self.device, self.lambda_gp
                    )

                    w_dist = c_real.mean() - c_fake.mean()
                    c_loss = -w_dist + gp
                    c_loss.backward()
                    self.opt_C.step()

                # ── Train Generator × 1 ───────────────────────────────
                self.opt_G.zero_grad()

                z    = torch.randn(n, self.latent_dim).to(self.device)
                fake = self.G(z, labels)
                g_loss = -self.C(fake, labels).mean()
                g_loss.backward()
                self.opt_G.step()

                epoch_c  += c_loss.item()
                epoch_g  += g_loss.item()
                epoch_w  += w_dist.item()
                n_batches += 1

                pbar.set_postfix({
                    "C": f"{c_loss.item():.3f}",
                    "G": f"{g_loss.item():.3f}",
                    "W": f"{w_dist.item():.3f}",
                })

            avg_c = epoch_c / n_batches
            avg_g = epoch_g / n_batches
            avg_w = epoch_w / n_batches

            self.c_losses.append(avg_c)
            self.g_losses.append(avg_g)
            self.w_distances.append(avg_w)

            print(f"  Epoch {epoch:>3}/{self.epochs} | "
                  f"C Loss: {avg_c:.4f} | "
                  f"G Loss: {avg_g:.4f} | "
                  f"W Dist: {avg_w:.4f}")

            if epoch % self.save_interval == 0:
                self._save_image_grid(epoch)

            self._save_checkpoint(epoch, avg_g)

        # post-training
        self._save_loss_plot()

        self.G.eval()
        with torch.no_grad():
            final = self.G(self.fixed_noise, self.fixed_labels)

        torchvision.utils.save_image(
            final,
            os.path.join(self.assets_dir, "final_generated_grid.png"),
            nrow=8, normalize=True, value_range=(-1, 1)
        )
        self._save_labeled_grid(final)
        print(f"\nBest checkpoint → {self.checkpoint_dir}/best_generator.pth\n")

    def get_history(self) -> dict:
        return {
            "c_losses":    self.c_losses,
            "g_losses":    self.g_losses,
            "w_distances": self.w_distances,
        }
