# DDPM Ablation Study on CIFAR-10

## Overview

This repository presents an experimental study of **Denoising Diffusion Probabilistic Models (DDPMs)** on the CIFAR-10 dataset. Rather than focusing solely on implementing a diffusion model, the goal of this project was to investigate how different design choices affect generation quality under limited computational resources.

The experiments analyze the impact of:

* Linear vs Cosine noise schedules
* Exponential Moving Average (EMA)
* Single-head vs Multi-head attention
* Distributed FID evaluation using multiple GPUs

---

## Experimental Setup

* **Dataset:** CIFAR-10
* **Image Resolution:** 32 × 32
* **Framework:** PyTorch
* **Distributed Training/Evaluation:** Hugging Face Accelerate
* **Evaluation Metric:** Fréchet Inception Distance (FID)

---

## Experiments

| Model   | Noise Schedule | EMA | Attention   |     FID ↓ |
| ------- | -------------- | --- | ----------- | --------: |
| Model 1 | Linear         | No  | Single-head |    293.59 |
| Model 2 | Cosine         | No  | Single-head |     79.25 |
| Model 3 | Cosine         | Yes | Single-head | **56.71** |
| Model 4 | Cosine         | Yes | Multi-head  |     60.65 |

---

## Key Findings

### 1. Cosine schedules significantly improve sample quality.

Replacing the linear noise schedule with a cosine schedule reduced FID from **293.59 to 79.25**, suggesting that the choice of noise schedule has a substantial impact on training effectiveness.

### 2. EMA consistently improves generation quality.

Applying Exponential Moving Average further improved performance:

* 79.25 → 56.71 FID

This aligns with observations reported in diffusion literature.

### 3. Multi-head attention offered limited benefits.

Under the same training budget, multi-head attention produced a similar FID to single-head attention:

* Single-head EMA: 56.71
* Multi-head EMA: 60.65

This suggests that increased attention complexity may not provide significant gains for CIFAR-10 under constrained compute.

---

## Repository Structure

```text
ddpm-ablation-study/
│
├── src/                 # Reusable components
├── scripts/             # Training and evaluation entry points
├── requirements.txt
└── README.md
```

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train Models

Example:

```bash
accelerate launch scripts/train_linear.py
```

```bash
accelerate launch scripts/train_cosine.py
```

### Evaluate FID

```bash
accelerate launch scripts/evaluate_fid.py
```

### Generate Samples

```bash
python scripts/generate_samples.py
```

---

## Generated Samples

Sample generations from each model can be found in:

```text
assets/samples/
```

Example images:

* Linear schedule baseline
* Cosine schedule
* Cosine + EMA (single-head)
* Cosine + EMA (multi-head)

---

## Limitations

* Experiments were conducted exclusively on CIFAR-10 (32 × 32 resolution).
* FID was computed using 5,000 generated samples.
* Results may vary under larger models, higher resolutions, or longer training schedules.
* The goal of this work was to study the relative effects of design choices rather than achieve state-of-the-art performance.

---

## Future Work

Potential extensions include:

* Scaling to higher-resolution datasets
* Exploring classifier-free guidance
* Investigating DDIM sampling
* Evaluating larger attention architectures
* Performing FID evaluation with larger sample counts

---

## Acknowledgements

This project was inspired by the original DDPM and Improved DDPM papers and was implemented using PyTorch and Hugging Face Accelerate.
