"""
Image Generation Script
========================
Generate images from a trained GAN checkpoint.

Usage:
    python generate.py --model conditional \\
        --checkpoint checkpoints/conditional_gan/best_generator.pth

    python generate.py --model vanilla \\
        --checkpoint checkpoints/vanilla_gan/best_generator.pth \\
        --num_samples 64

Arguments:
    --model        : vanilla | conditional
    --checkpoint   : path to the .pth checkpoint file
    --num_samples  : images to generate per class (default: 8)
    --output_dir   : directory to save generated images
                     (default: assets/<model_type>/generated/)
    --seed         : random seed for reproducibility (default: 42)
    --device       : cpu | cuda | mps  (auto-detected by default)
"""

import argparse
import os
import sys
import torch
import torchvision

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── CIFAR-10 class names ────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird",  "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck",
]


# ── Argument parsing ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images from a trained GAN checkpoint.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vanilla", "conditional"],
        help=(
            "Which model architecture to load:\n"
            "  vanilla     — Vanilla GAN (MNIST, grayscale 28×28)\n"
            "  conditional — Conditional WGAN-GP (CIFAR-10, RGB 32×32)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pth checkpoint file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of images to generate (vanilla) or per class (conditional). Default: 8",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the generated images (default: assets/<model>/generated/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto-detect)",
    )
    return parser.parse_args()


# ── Device selection ─────────────────────────────────────────────────────────
def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Checkpoint loader ────────────────────────────────────────────────────────
def load_checkpoint(path: str, device: torch.device) -> dict:
    if not os.path.exists(path):
        print(f"\n[ERROR] Checkpoint not found: {path}", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


# ── Conditional GAN generator ────────────────────────────────────────────────
def generate_conditional(
    ckpt: dict,
    device: torch.device,
    num_samples: int,
    output_dir: str,
    seed: int,
) -> None:
    # FIX: class is ConditionalGenerator, import path matches src/conditional_gan/generator.py
    from src.conditional_gan.generator import ConditionalGenerator

    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg  = cfg.get("data",  {})

    latent_dim    = model_cfg.get("latent_dim",    100)
    embedding_dim = model_cfg.get("embedding_dim", 100)
    base_channels = model_cfg.get("base_channels", 256)
    num_classes   = data_cfg.get("num_classes",    10)
    img_channels  = data_cfg.get("image_channels", 3)

    G = ConditionalGenerator(
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        base_channels=base_channels,
        image_channels=img_channels,
    ).to(device)

    G.load_state_dict(ckpt["g_state_dict"])
    G.eval()

    print(f"  Generator      : {G.count_parameters():,} params")
    print(f"  Latent dim     : {latent_dim}")
    print(f"  Classes        : {num_classes}")
    print(f"  Samples/class  : {num_samples}")

    torch.manual_seed(seed)
    noise_list, label_list = [], []
    for cls in range(num_classes):
        noise_list.append(torch.randn(num_samples, latent_dim))
        label_list.append(torch.full((num_samples,), cls, dtype=torch.long))

    noise  = torch.cat(noise_list).to(device)
    labels = torch.cat(label_list).to(device)

    with torch.no_grad():
        fake = G(noise, labels)   # (num_classes * num_samples, 3, 32, 32)

    os.makedirs(output_dir, exist_ok=True)

    # ── Plain grid ─────────────────────────────────────────────────────────
    grid_path = os.path.join(output_dir, "generated_grid.png")
    torchvision.utils.save_image(
        fake, grid_path, nrow=num_samples, normalize=True, value_range=(-1, 1)
    )
    print(f"\n  ✓ Grid saved        → {grid_path}")

    # ── Labeled grid (class names on left) ─────────────────────────────────
    imgs = (fake.cpu().clamp(-1, 1) + 1) / 2.0  # → [0, 1]

    fig, axes = plt.subplots(
        num_classes, num_samples,
        figsize=(num_samples * 1.8, num_classes * 1.8),
    )

    for cls in range(num_classes):
        for col in range(num_samples):
            img = imgs[cls * num_samples + col].permute(1, 2, 0).numpy()
            axes[cls, col].imshow(img)
            axes[cls, col].axis("off")
        axes[cls, 0].set_ylabel(
            CIFAR10_CLASSES[cls], fontsize=10,
            rotation=0, labelpad=60, va="center",
        )

    plt.suptitle(
        f"Conditional WGAN-GP — Generated CIFAR-10 Images  "
        f"(epoch {ckpt.get('epoch', '?')})",
        fontsize=13,
    )
    plt.tight_layout()
    labeled_path = os.path.join(output_dir, "generated_labeled_grid.png")
    plt.savefig(labeled_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Labeled grid saved → {labeled_path}")


# ── Vanilla GAN generator ────────────────────────────────────────────────────
def generate_vanilla(
    ckpt: dict,
    device: torch.device,
    num_samples: int,
    output_dir: str,
    seed: int,
) -> None:
    # FIX: class is Generator (not VanillaGenerator) per src/vanilla_gan/generator.py
    from src.vanilla_gan.generator import Generator

    cfg       = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    latent_dim  = model_cfg.get("latent_dim", 100)
    image_dim   = model_cfg.get("image_dim",  784)

    # FIX: config key is 'hidden_dims' (list), not 'hidden_dim' (scalar)
    hidden_dims = model_cfg.get("hidden_dims", [256, 512, 1024])

    G = Generator(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        image_dim=image_dim,
    ).to(device)

    G.load_state_dict(ckpt["g_state_dict"])
    G.eval()

    print(f"  Generator      : {G.count_parameters():,} params")
    print(f"  Latent dim     : {latent_dim}")
    print(f"  Samples        : {num_samples}")

    torch.manual_seed(seed)
    noise = torch.randn(num_samples, latent_dim).to(device)

    with torch.no_grad():
        fake = G(noise)   # (N, 1, 28, 28)

    os.makedirs(output_dir, exist_ok=True)

    nrow = min(8, num_samples)
    grid_path = os.path.join(output_dir, "generated_grid.png")
    torchvision.utils.save_image(
        fake, grid_path, nrow=nrow, normalize=True, value_range=(-1, 1)
    )
    print(f"\n  ✓ Grid saved → {grid_path}")

    # ── Annotated matplotlib figure ────────────────────────────────────────
    imgs   = (fake.cpu().clamp(-1, 1) + 1) / 2.0  # → [0, 1]
    ncols  = nrow
    nrows  = (num_samples + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    imgs_np = imgs.squeeze(1).numpy()  # (N, 28, 28)
    for ax, img in zip(axes_flat, imgs_np):
        ax.imshow(img, cmap="gray_r")
        ax.axis("off")
    for ax in axes_flat[len(imgs_np):]:
        ax.axis("off")

    plt.suptitle(
        f"Vanilla GAN — Generated MNIST Digits  (epoch {ckpt.get('epoch', '?')})",
        fontsize=12,
    )
    plt.tight_layout()
    annotated_path = os.path.join(output_dir, "generated_annotated.png")
    plt.savefig(annotated_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Annotated figure → {annotated_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    device = get_device(args.device)

    output_dir = args.output_dir or os.path.join(
        "assets", f"{args.model}_gan", "generated"
    )

    print(f"\n{'='*55}")
    print(f"  GAN Image Generator")
    print(f"{'='*55}")
    print(f"  Model          : {args.model}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Device         : {device}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Seed           : {args.seed}")

    ckpt = load_checkpoint(args.checkpoint, device)
    print(f"  Checkpoint epoch : {ckpt.get('epoch', 'unknown')}")

    if args.model == "conditional":
        generate_conditional(
            ckpt=ckpt,
            device=device,
            num_samples=args.num_samples,
            output_dir=output_dir,
            seed=args.seed,
        )
    elif args.model == "vanilla":
        generate_vanilla(
            ckpt=ckpt,
            device=device,
            num_samples=args.num_samples,
            output_dir=output_dir,
            seed=args.seed,
        )

    print(f"\n{'='*55}")
    print("  Generation complete!")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
