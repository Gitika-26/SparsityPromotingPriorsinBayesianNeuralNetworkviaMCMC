# A Study of Sparsity-Promoting Priors in Bayesian Neural Networks

## 📌 Overview

This repository contains the code and experimental setup for the project titled  
**"A Study of Sparsity-Promoting Priors in Bayesian Neural Networks"**  
presented by **Gitika** (IISER Tirupati)  
under the supervision of **Dr. Sandipan Karmakar** (MNIT Jaipur).

The study investigates how different **sparsity-promoting priors** affect the performance and weight sparsity of Bayesian Neural Networks (BNNs), using **Markov Chain Monte Carlo (MCMC)** with **Langevin dynamics**.

## 🧠 Neural Network Architecture

- Type: Feedforward Neural Network
- Layers: Input → Hidden (1–2 layers, 16–64 units) → Output
- Activations: ReLU or Tanh
- Output Activation:
  - Linear for regression
  - Softmax for classification

---

## 📊 Datasets

- **Regression**: Boston Housing Dataset  
  [Download](https://encr.pw/iCARd)

- **Classification**: Mobile Price Range Dataset  
  [Download](https://encr.pw/I2Kvg)

---

## 🔍 Priors Used

### 🟦 Gaussian Prior (Baseline)

$$
w \sim \mathcal{N}(0, \sigma^2)
$$

### 🟨 Horseshoe Prior

$$
w \sim \mathcal{N}(0, \tau^2 \lambda^2), \quad \lambda \sim \text{C}^+(0,1), \quad \tau \sim \text{C}^+(0,1)
$$

- \( \text{C}^+ \): Half-Cauchy distribution  
- \( \lambda \): Local shrinkage  
- \( \tau \): Global shrinkage  

### 🟥 Spike-and-Slab Prior

$$
p(w) = \pi \cdot \delta(w) + (1 - \pi) \cdot \mathcal{N}(w \mid 0, \sigma^2)
$$

- \( \delta(w) \): Dirac delta function (spike at zero)  
- \( \pi \): Probability of zero weights  
- \( \sigma^2 \): Variance of the slab

---

## 🔁 Bayesian Inference: Langevin MCMC

- We use **MCMC with Langevin Dynamics** for posterior sampling.
- Gradients are used to guide the weight sampling.
- Chains are run per prior on each dataset.
- Early stopping based on validation performance.

---

## 🎯 Evaluation Metrics

- **Regression**: Mean Squared Error (MSE)
- **Classification**: Accuracy and F1 Score
- Final predictions are averaged across posterior samples.

---

## 🛠 Libraries Used

- `NumPy`
- `SciPy`
- `Pandas`
- `Matplotlib`
- `Seaborn`
- `Scikit-learn`
- `tqdm`

  ## 🧪 Results

### 🔹 Regression Task (Boston Housing Dataset)

| Prior              | Mean Squared Error (MSE) | Observations                                             |
|--------------------|--------------------------|----------------------------------------------------------|
| Gaussian (Baseline)| 15.72                    | Higher variance in predictions; moderate overfitting     |
| Horseshoe          | 11.30                    | Better generalization with many weights driven to zero   |
| Spike-and-Slab     | 10.85                    | Most sparse and accurate; pruned irrelevant weights      |

---

### 🔸 Classification Task (Mobile Price Range Dataset)

| Prior              | Accuracy (%) | F1 Score | Observations                                               |
|--------------------|--------------|----------|------------------------------------------------------------|
| Gaussian (Baseline)| 85.2         | 0.851    | Overfits on some features; limited feature selection       |
| Horseshoe          | 88.7         | 0.883    | Robust performance; adapts well to feature importance      |
| Spike-and-Slab     | 89.5         | 0.891    | Strong feature pruning; highly sparse and interpretable    |

---

### 💡 Key Insights

- **Horseshoe Prior** enforces *global and local shrinkage* through its hierarchical half-Cauchy structure. It allows a few weights to remain large (capturing important features) while aggressively shrinking the rest, thus improving generalization without sacrificing expressiveness.

- **Spike-and-Slab Prior** explicitly models weights as a mixture of a delta function (spike at zero) and a Gaussian (slab), making it the most interpretable and sparse prior. It excels at performing *automatic feature selection* by entirely zeroing out irrelevant weights.

- **Gaussian Prior**, while simple and computationally efficient, applies uniform shrinkage to all weights. This limits its ability to differentiate between important and redundant features, leading to poorer generalization and lower sparsity.

---

These results demonstrate that **incorporating structured sparsity priors** into Bayesian Neural Networks significantly improves both predictive performance and model compactness, especially in high-dimensional or noisy data scenarios.

## 📚 References
R. Chandra and J. Simmons, Bayesian Neural Networks via MCMC: A Python-Based Tutorial, arXiv:2203.12557 (2022).

S. van Erp, Shrinkage Priors for Bayesian Penalized Regression, arXiv:1505.02646 (2015).

Z. Lu and W. Lou, Bayesian Approaches to Variable Selection: A Comparative Study, arXiv:2307.03481 (2023).
