# Time-Series-FMoE-Copula
A deep probabilistic forecasting framework for ultra-high-dimensional time series using 4-bit LLMs, Frequency-routed MoE, and memory-optimized Gaussian Copula. Solves OOM for 800+ variables.
# 🚀 FMoE-Copula: Ultra-High-Dimensional Time Series Forecasting with LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-Latest-orange.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FMoE-Copula** is a state-of-the-art deep probabilistic forecasting framework designed for ultra-high-dimensional and non-stationary time series. By synergizing 4-bit quantized Large Language Models (LLMs) with Frequency-routed Mixture-of-Experts (MoE) and a highly optimized Gaussian Copula joint probability mechanism, this project achieves near point forecasting (Autoformer, 2021) while providing rigorous high-dimensional uncertainty estimation.
# ===============================================
in_length=96,pre_length=96 \
Wheather: Ours                  -- MSE: 0.172  MAE: 0.221  \
          DUET (KDD2025)        -- MSE: 0.146  MAE: 0.191 \
          FreEformer(IJCAI2025) --  MSE: 0.239  MAE: 0.260 \
          
in_length=96,pre_length=96 \
Exchange: Ours                  -- MSE: 0.100  MAE: 0.224  \
          DUET (KDD2025)        -- MSE: 0.080  MAE: 0.198  \
          FreEformer(IJCAI2025) -- MSE: 0.354  MAE: 0.399 \
          
in_length=96,pre_length=96 \
Traffic:  Ours                  -- MSE:   MAE:   \
          DUET (KDD2025)        -- MSE:   MAE:   \
          FreEformer(IJCAI2025) -- MSE:   MAE:  \
          
in_length=96,pre_length=96 \
Electricity: Ours               -- MSE:   MAE:   \
          DUET (KDD2025)        -- MSE:   MAE:   \
          FreEformer(IJCAI2025) -- MSE:   MAE:  \
          
in_length=36,pre_length=24 \
ILL:      Ours                  -- MSE: 1.531  MAE: 0.842 \
          DUET (KDD2025)        -- MSE: 1.906  MAE: 0.835 \
          FreEformer(IJCAI2025) -- MSE: 1.577  MAE: 0.760 \
# ===============================================
## ✨ Core Innovations 

[cite_start]This repository is engineered to survive the "Curse of Dimensionality", easily scaling up to **800+ variables** (e.g., the Traffic dataset [cite: 224]) on a single GPU without Out-Of-Memory (OOM) errors.

* **🧠 Frequency-Routed MoE-LoRA**: Employs Fast Fourier Transform (FFT) to extract frequency domain features, dynamically routing tokens to specialized LoRA experts embedded within the frozen LLM's attention layers.
* **📉 Reversible Instance Normalization (RevIN)**: Effectively handles non-stationary shifts and anomalies in long-tail data, anchoring the Mean Squared Error (MSE) from exploding.
* **⚡ $O(D^2)$ Ultra-Fast Copula Loss**: Completely abandons the computationally expensive and unstable matrix inversion in Marginal NLL and Copula NLL. Implements a highly optimized `torch.linalg.solve_triangular` approach to solve quadratic forms, reducing memory consumption by over 90% and preventing `NaN` gradient collapses.
* **🌌 Dimension-Adaptive Architecture**: 
  * *Low-Dim (e.g., Exchange 8D)*: Dynamically unfolds Time-Variant Copula structures.
  * *High-Dim (e.g., Traffic 862D)*: Collapses the time dimension for Static Copula, modeling stable physical topological correlations while saving massive VRAM.
* **🎲 Memory-Efficient Reparameterization MC Sampling**: Bypasses the 126GB broadcasting black hole of PyTorch's native `MultivariateNormal` by implementing raw Tensor-level `torch.einsum` operations for Monte Carlo sampling.
* **🛡️ Two-Stage Decoupling Training**: A curriculum learning strategy that forces the LLM to learn deterministic trajectories via MSE first, before detaching gradients to exclusively fit the complex Copula noise distributions.

## 📊 Supported Datasets
[cite_start]We follow the standard evaluation protocols from the long-term time series forecasting benchmark[cite: 218].
* **Low/Mid Dimensional**: ILI, Exchange (8D), Weather (21D), ETT (7D)
* **Ultra-High Dimensional**: Electricity (321D), Traffic (862D)

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/FMoE-Copula.git](https://github.com/jialei8788-source/FMoE-Copula.git)
cd FMoE-Copula

# Create conda environment
conda create -n fmoe_copula python=3.10
conda activate fmoe_copula

# Install dependencies (ensure you have the correct CUDA version for PyTorch)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers bitsandbytes accelerate
pip install numpy pandas matplotlib seaborn tqdm
