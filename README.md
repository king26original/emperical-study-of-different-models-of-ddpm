# DDPM Ablation Study on CIFAR-10

## Overview

This repository presents an experimental study of **Denoising Diffusion Probabilistic Models (DDPMs)** on the CIFAR-10 dataset. Rather than focusing solely on implementing a diffusion model, the goal of this project was to investigate how different design choices affect generation quality under limited computational resources.

The experiments analyze the impact of:

* Linear vs Cosine noise schedules
* Exponential Moving Average (EMA)
* Single-head vs Multi-head attention
* Distributed FID evaluation using multiple GPUs

---

## Introduction to Denoising Diffusion Probabilistic Models (DDPMs)

Denoising Diffusion Probabilistic Models (DDPMs) are a class of generative models that learn to synthesize realistic data by gradually reversing a noise corruption process. Unlike Generative Adversarial Networks (GANs), which generate samples in a single forward pass, DDPMs generate data through a sequence of iterative denoising steps.

The central idea behind DDPMs is surprisingly simple:

1. **Forward Diffusion Process:**
   Starting from a real image, small amounts of Gaussian noise are progressively added over many timesteps until the image becomes indistinguishable from pure noise.

2. **Reverse Denoising Process:**
   A neural network is trained to predict and remove the noise added at each timestep. Once trained, the model can start from random Gaussian noise and iteratively denoise it to generate new samples.

### Forward Process

The forward diffusion process transforms a clean image (x_0) into increasingly noisy versions (x_t):

[
q(x_t \mid x_{t-1}) = \mathcal{N}\left(
\sqrt{1-\beta_t},x_{t-1},
\beta_t I
\right)
]

where:

* (x_0) is the original image,
* (x_t) is the noisy image at timestep (t),
* (\beta_t) controls the amount of noise added at each step.

After a sufficiently large number of timesteps, the data distribution approaches a standard Gaussian distribution.

### Reverse Process

The reverse process attempts to undo the corruption introduced by the forward process:

[
p_\theta(x_{t-1}\mid x_t)
]

A neural network parameterized by (\theta) is trained to estimate the noise present in (x_t). By repeatedly applying this learned denoising process, the model gradually reconstructs a clean image from random noise.

### Training Objective

Instead of directly predicting the clean image, DDPMs are commonly trained to predict the noise that was added:

[
L_{\text{simple}}
=================

\mathbb{E}
\left[
\left|
\epsilon
--------

\epsilon_\theta(x_t,t)
\right|^2
\right]
]

where:

* (\epsilon) is the true Gaussian noise,
* (\epsilon_\theta(x_t,t)) is the predicted noise,
* the objective minimizes the mean squared error between the two.

### Why DDPMs?

Diffusion models have gained significant attention because they offer several advantages:

* Stable training compared to adversarial approaches.
* High-quality and diverse sample generation.
* A principled probabilistic formulation.
* Flexibility to incorporate architectural and sampling improvements.

However, their iterative generation process can be computationally expensive, making design choices such as noise schedules, Exponential Moving Average (EMA), and attention mechanisms particularly important in practice.

### Focus of This Study

Rather than proposing a new diffusion architecture, this work investigates the practical impact of several widely used DDPM design choices. Specifically, we examine how different noise schedules, EMA, and attention mechanisms influence generation quality on CIFAR-10 under limited computational resources.

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

## Training Dynamics

### Model 1: Linear + Single-head
### Loss Curve
<img width="1000" height="500" alt="1" src="https://github.com/user-attachments/assets/2f2e8e5b-e8d1-4310-942d-c0480b3e2d5b" />




### Model 2: Cosine + Single-head
### Loss Curve
<img width="1000" height="500" alt="2" src="https://github.com/user-attachments/assets/8feca5c8-9da9-41a6-8354-9949d13fa3ea" />




### Model 3: Cosine + EMA + Single-head
### Loss Curve
<img width="1000" height="500" alt="3" src="https://github.com/user-attachments/assets/91f202fc-a9e0-43ed-92ec-c1667b3e135f" />




### Model 4: Cosine + EMA + Multi-head
### Loss Curve
<img width="1000" height="500" alt="4" src="https://github.com/user-attachments/assets/8da4d5a7-f4ed-4517-9ccd-514b0f19ec2a" />

---

## Generated Samples
### Model 1: Linear + Single-head
<img width="1617" height="176" alt="image" src="https://github.com/user-attachments/assets/11fb3b1c-2765-4f65-956c-a7c7962e47c4" />
<img width="1615" height="174" alt="image" src="https://github.com/user-attachments/assets/7fee6a8e-5616-46fb-92dd-5e43e4f9b23a" />
<img width="1666" height="180" alt="image" src="https://github.com/user-attachments/assets/e356a07a-9c49-44fb-a5d3-1e1241299a84" />

### Model 2: Cosine + Single-head
<img width="1700" height="183" alt="image" src="https://github.com/user-attachments/assets/a74a9dba-dc22-4ebf-89da-e3b1b6a8d728" />
<img width="1614" height="174" alt="image" src="https://github.com/user-attachments/assets/59cfc59e-d461-4b26-b784-eacf8a590a69" />
<img width="1111" height="120" alt="image" src="https://github.com/user-attachments/assets/467e3fce-0ad2-4824-9573-08d4f038837f" />

### Model 3: Cosine + EMA + Single-head
<img width="1112" height="120" alt="image" src="https://github.com/user-attachments/assets/d810551d-2b9b-4172-bdb2-e984acc19237" />
<img width="1112" height="121" alt="image" src="https://github.com/user-attachments/assets/3abe7483-888f-4f64-80d6-c6b4421a7aa5" />
<img width="1109" height="118" alt="image" src="https://github.com/user-attachments/assets/12839280-6c36-47c6-b63a-298e156cdfdf" />
<img width="1111" height="120" alt="image" src="https://github.com/user-attachments/assets/3f2bfd13-450e-4381-9d50-4f365f2ba7b8" />

### Model 4: Cosine + EMA + Multi-head
<img width="1110" height="120" alt="image" src="https://github.com/user-attachments/assets/c53b4723-8dc1-4c0b-91f7-4822f19ed90a" />
<img width="1108" height="119" alt="image" src="https://github.com/user-attachments/assets/afb2e1cc-9f2f-48e8-b9fa-30e19329b709" />
<img width="1107" height="116" alt="image" src="https://github.com/user-attachments/assets/4bbaf10a-648e-46a0-b037-70593ca93cac" />


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

---

## Takeaways

Under constrained compute, noise schedules and EMA mattered far more than increasing attention complexity. This project demonstrates how systematic experimentation can reveal which diffusion model components provide the greatest practical benefit.
