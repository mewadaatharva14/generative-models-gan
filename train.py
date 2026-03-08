"""
Training Entry Point
=====================
Train either Vanilla GAN on MNIST or Conditional WGAN-GP on CIFAR-10.

Usage:
    python train.py --model vanilla
    python train.py --model conditional

Arguments:
    --model : vanilla | conditional
    --config: optional path to config file
              defaults to configs/vanilla_gan_config.yaml
                      or configs/conditional_gan_config.yaml
"""

import argparse
import os
import yaml


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GAN model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vanilla", "conditional"],
        help=(
            "Model to train:\n"
            "  vanilla     — Vanilla GAN on MNIST\n"
            "  conditional — Conditional WGAN-GP on CIFAR-10"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config file.\n"
            "Defaults to configs/vanilla_gan_config.yaml\n"
            "         or configs/conditional_gan_config.yaml"
        ),
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------

DEFAULT_CONFIGS = {
    "vanilla":     "configs/vanilla_gan_config.yaml",
    "conditional": "configs/conditional_gan_config.yaml",
}


def load_config(model: str, config_path: str | None) -> dict:
    path = config_path or DEFAULT_CONFIGS[model]

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nConfig file not found: {path}\n"
            f"Expected at: {DEFAULT_CONFIGS[model]}\n"
            f"Run from the repo root directory."
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.model, args.config)

    print(f"\n{'='*55}")
    print(f"  Model  : {args.model}")
    print(f"  Config : {args.config or DEFAULT_CONFIGS[args.model]}")
    print(f"{'='*55}")

    if args.model == "vanilla":
        from src import VanillaGANTrainer
        trainer = VanillaGANTrainer(config)

    elif args.model == "conditional":
        from src import ConditionalGANTrainer
        trainer = ConditionalGANTrainer(config)

    trainer.train()


if __name__ == "__main__":
    main()