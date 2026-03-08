# 🎨 Generative Models — Vanilla GAN & Conditional WGAN-GP

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-00C28B?style=flat)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-mewadaatharva14-181717?style=flat&logo=github)](https://github.com/mewadaatharva14)

> Two GAN implementations from scratch in PyTorch — a Vanilla GAN on MNIST
> and a Conditional WGAN-GP on CIFAR-10 with class-controlled image generation.

---

## 📌 Overview

This repository implements two progressively complex generative adversarial
networks. The Vanilla GAN demonstrates the core GAN training loop — two
networks competing in a minimax game. The Conditional WGAN-GP extends this
with Wasserstein loss, gradient penalty, and label conditioning via learned
embeddings — allowing explicit control over which class of image is generated.
Both models are trained end-to-end with clean, configurable PyTorch code.

---

## 🗂️ Project Structure

```
generative-models-gan/
├── src/
│   ├── vanilla_gan/
│   │   ├── generator.py        ← FC Generator: z(100) → 784 + BN + LeakyReLU + Tanh
│   │   ├── discriminator.py    ← FC Discriminator: 784 → 1 + LeakyReLU + Dropout
│   │   └── trainer.py          ← BCELoss training loop, label smoothing, checkpoints
│   │
│   └── conditional_gan/
│       ├── generator.py        ← ConvTranspose2d Generator + nn.Embedding label conditioning
│       ├── discriminator.py    ← Conv2d Critic + SpectralNorm + label channel injection
│       ├── gradient_penalty.py ← WGAN-GP penalty: interpolation + gradient norm constraint
│       └── trainer.py          ← Wasserstein loop, critic×N updates, labeled grid saving
│
├── notebooks/
│   ├── 01_vanilla_gan_mnist.ipynb
│   └── 02_conditional_wgan_gp_cifar10.ipynb
│
├── configs/
│   ├── vanilla_gan_config.yaml
│   └── conditional_gan_config.yaml
│
├── assets/                     ← committed plots and final image grids
├── samples/                    ← per-epoch generated images (gitignored)
├── checkpoints/                ← saved model weights (gitignored)
├── data/                       ← datasets downloaded automatically
├── .gitignore
├── requirements.txt
├── generate.py                 ← inference script — load checkpoint, generate images
├── README.md
└── train.py                    ← unified entry point
```

---

## 🏗️ Models

### 1 · Vanilla GAN — MNIST Digit Generation
**Dataset:** MNIST — 60,000 grayscale images, 28×28  
**Task:** Generate realistic handwritten digits from random noise

#### The GAN Framework

| Network | Input | Output | Goal |
|---|---|---|---|
| Generator G | Noise z ~ N(0,1) | Fake image | Fool D |
| Discriminator D | Real or fake image | P(real) ∈ (0,1) | Detect fakes |

#### Objective Functions

**Discriminator** maximizes:
$$\mathcal{L}_D = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$$

**Generator** minimizes (non-saturating variant):
$$\mathcal{L}_G = -\mathbb{E}[\log D(G(z))]$$

#### Architecture

| Layer | Operation | Output Shape |
|---|---|---|
| Input | Noise vector | (B, 100) |
| Linear + BN + LeakyReLU | z → 256 | (B, 256) |
| Linear + BN + LeakyReLU | 256 → 512 | (B, 512) |
| Linear + BN + LeakyReLU | 512 → 1024 | (B, 1024) |
| Linear + Tanh | 1024 → 784 | (B, 784) |
| Reshape | 784 → image | (B, 1, 28, 28) |

#### Results

| Metric | Value |
|---|---|
| Final D Loss | — |
| Final G Loss | — |
| Training Epochs | 5 (CPU) / 50+ (GPU) |

> Run `python train.py --model vanilla` to fill this table.

---

### 2 · Conditional WGAN-GP — CIFAR-10 Class-Controlled Generation
**Dataset:** CIFAR-10 — 50,000 RGB images, 32×32, 10 classes  
**Task:** Generate images conditioned on a class label

#### Vanilla GAN → WGAN-GP

| Property | Vanilla GAN | WGAN-GP |
|---|---|---|
| Loss | Binary Cross-Entropy | Wasserstein distance |
| Output activation | Sigmoid → probability | None → raw score |
| Lipschitz constraint | None | Gradient Penalty |
| Critic updates per G step | 1 | 5 (standard) |
| Training stability | Often unstable | Significantly more stable |

#### Wasserstein Loss

**Critic** minimizes:
$$\mathcal{L}_C = \underbrace{\mathbb{E}[C(\tilde{x}, y)]}_{\text{fake score}} - \underbrace{\mathbb{E}[C(x, y)]}_{\text{real score}} + \underbrace{\lambda \cdot GP}_{\text{gradient penalty}}$$

**Generator** minimizes:
$$\mathcal{L}_G = -\mathbb{E}[C(G(z, y),\ y)]$$

#### Gradient Penalty

Enforces 1-Lipschitz constraint on the critic:

$$GP = \lambda \cdot \mathbb{E}_{\hat{x}}\left[(\|\nabla_{\hat{x}} C(\hat{x}, y)\|_2 - 1)^2\right]$$

where $\hat{x} = \epsilon \cdot x_{\text{real}} + (1-\epsilon) \cdot x_{\text{fake}}$, $\epsilon \sim \mathcal{U}(0,1)$

#### Label Conditioning via nn.Embedding

```
Generator:  [z (100) | embed(label) (100)] → concat (200) → Linear → ConvTranspose2d blocks
Critic:      embed(label) → project to (1, 32, 32) → concat with image (3+1, 32, 32)
```

This forces the critic to evaluate whether the **image matches its label**,
not just whether it looks realistic in isolation.

#### Architecture

| Layer | Operation | Output Shape |
|---|---|---|
| Input | z + label embed concat | (B, 200) |
| Linear + LeakyReLU | 200 → 256×4×4 | (B, 4096) |
| Reshape | → feature map | (B, 256, 4, 4) |
| ConvTranspose2d + BN + LeakyReLU | upsample ×2 | (B, 128, 8, 8) |
| ConvTranspose2d + BN + LeakyReLU | upsample ×2 | (B, 64, 16, 16) |
| ConvTranspose2d + Tanh | upsample ×2 | (B, 3, 32, 32) |

#### Results

| Metric | Value |
|---|---|
| Final Critic Loss | — |
| Final Generator Loss | — |
| Final Wasserstein Distance | — |
| Training Epochs | 3 (CPU) / 50+ (GPU) |

> Run `python train.py --model conditional` to fill this table.

---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/mewadaatharva14/generative-models-gan.git
cd generative-models-gan
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train**
```bash
# Vanilla GAN on MNIST
python train.py --model vanilla

# Conditional WGAN-GP on CIFAR-10
python train.py --model conditional
```

Datasets download automatically on first run.  
Loss curves and final image grids saved to `assets/` after training.  
Per-epoch samples saved to `samples/` (gitignored).

**5. Generate images from a trained checkpoint**
```bash
# Vanilla GAN — 64 random images
python generate.py --model vanilla \
    --checkpoint checkpoints/vanilla_gan/best_generator.pth

# Conditional GAN — all 10 classes, 8 images each
python generate.py --model conditional \
    --checkpoint checkpoints/conditional_gan/best_generator.pth

# Conditional GAN — specific classes only
python generate.py --model conditional \
    --checkpoint checkpoints/conditional_gan/best_generator.pth \
    --classes 0 1 5
```

---

## 📓 Notebooks

| Notebook | Dataset | What it shows |
|---|---|---|
| `01_vanilla_gan_mnist.ipynb` | MNIST | Model architecture · Loss curves · Generated grid · Epoch progression |
| `02_conditional_wgan_gp_cifar10.ipynb` | CIFAR-10 | Loss curves · Per-class labeled grid · Epoch progression |

```bash
jupyter notebook notebooks/
```

---

## 🔑 Key Implementation Details

**Why `fake.detach()` in the Discriminator step:**  
During D's backward pass, gradients should only update D's weights — not G's.
Without `.detach()`, PyTorch would propagate gradients all the way through G
during D's update, wastefully computing G's gradients when G is not being updated.

**Why label smoothing 0.9 instead of 1.0:**  
If D is trained with hard labels (1.0), it becomes overconfident — assigning
near-zero probability to fake images. This gives G near-zero gradient, making
it impossible to learn. Smoothing to 0.9 keeps D slightly uncertain and
maintains a learning signal for G throughout training.

**Why `beta1=0.0` in WGAN-GP Adam:**  
Standard Adam uses `beta1=0.9` — momentum from past gradients. In WGAN-GP,
the critic's gradient direction changes frequently as the real/fake distributions
evolve. Momentum from stale gradients destabilizes the Wasserstein estimate.
`beta1=0.0` removes momentum entirely, keeping critic updates responsive.

**Why `create_graph=True` in gradient penalty:**  
The gradient penalty is computed from gradients of the critic w.r.t.
interpolated inputs. When the penalty is itself differentiated during the
critic's backward pass, PyTorch needs a second-order gradient. `create_graph=True`
keeps the computation graph alive for this second differentiation.

**Why SpectralNorm on Critic layers:**  
SpectralNorm divides each weight matrix by its largest singular value,
bounding the Lipschitz constant of each layer. This works alongside the
gradient penalty to enforce the 1-Lipschitz constraint required for
stable Wasserstein distance estimation.

**Why `drop_last=True` in CIFAR-10 DataLoader:**  
The gradient penalty computes gradients w.r.t. interpolated images using
`torch.autograd.grad`. This requires a consistent batch size. A partial
last batch would cause shape mismatches in the interpolation step.

---

## 📚 References

| Resource | Link |
|---|---|
| Original GAN Paper | [Goodfellow et al. 2014](https://arxiv.org/abs/1406.2661) |
| WGAN Paper | [Arjovsky et al. 2017](https://arxiv.org/abs/1701.07875) |
| WGAN-GP Paper | [Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028) |
| DCGAN Paper | [Radford et al. 2015](https://arxiv.org/abs/1511.06434) |
| MNIST Dataset | [LeCun et al.](http://yann.lecun.com/exdb/mnist/) |
| CIFAR-10 Dataset | [Krizhevsky 2009](https://www.cs.toronto.edu/~kriz/cifar.html) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with 🧠 by <a href="https://github.com/mewadaatharva14">mewadaatharva14</a>
</p>
