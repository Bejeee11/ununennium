# Uncertainty Quantification in Geospatial AI

## 1. The Epistemic and Aleatoric Dichotomy

In safety-critical EO applications (flood mapping, deforestation monitoring), a point prediction $\hat{y}$ is insufficient. We require a full predictive distribution $P(y|x)$. Uncertainty is formally decomposed into two orthogonal components:

### 1.1 Aleatoric Uncertainty (Data Uncertainty)
Captures noise inherent in the observations.
*   **Source:** Sensor noise, atmospheric scattering, labeling ambiguity (e.g., transition zones between forest and shrub).
*   **Properties:** Irreducible with more data.
*   **Modeling:** Learned by the loss function (e.g., Heteroscedastic Regression).

### 1.2 Epistemic Uncertainty (Model Uncertainty)
Captures ignorance about the model parameters.
*   **Source:** Lack of training data in a specific domain, distribution shift (Covariate Shift).
*   **Properties:** Reducible with more data.
*   **Modeling:** Bayesian Ensembles, Monte Carlo Dropout.

---

## 2. Bayesian Deep Learning Methods

### 2.1 Monte Carlo (MC) Dropout

Approximates a Deep Gaussian Process. During inference, dropout is kept active, and $T$ stochastic forward passes are performed.

**Prediction:**
$$ \mathbb{E}[y] \approx \frac{1}{T} \sum_{t=1}^T \hat{y}_t $$

**Uncertainty (Entropy):**
$$ H(y|x) = - \sum_{c} p_{avg}(c) \log p_{avg}(c) $$

Where $p_{avg}$ is the averaged probability across $T$ passes.

### 2.2 Deep Ensembles

Training $M$ independent models with random initialization and shuffled data batches. This is the **gold standard** for uncertainty quality but computationally expensive.

**Total Variance:**
$$ \text{Var}(y) = \underbrace{\frac{1}{M} \sum_{m=1}^M \text{Var}(y_m)}_{\text{Aleatoric}} + \underbrace{\frac{1}{M} \sum_{m=1}^M (\mu_m - \bar{\mu})^2}_{\text{Epistemic}} $$

### 2.3 Evidential Deep Learning (EDL)

Ununennium implements EDL, which places a Dirichlet distribution over the class probabilities. The network predicts the *parameters* $\alpha$ of the Dirichlet distribution rather than the probabilities directly.

**Confidence:**
$$ \text{Evidence } e_k = f(x)_k $$
$$ \alpha_k = e_k + 1 $$
$$ S = \sum \alpha_k $$
$$ \text{Uncertainty } u = \frac{K}{S} $$

This allows single-pass deterministic uncertainty estimation.

---

## 3. Metrics for Uncertainty

### 3.1 Expected Calibration Error (ECE)

Measures if confidence matches accuracy. A perfectly calibrated model with 80% confidence should be correct 80% of the time.

$$ \text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} | \text{acc}(B_m) - \text{conf}(B_m) | $$

Where $B_m$ are bins of confidence scores.

### 3.2 Brier Score

Proper scoring rule that measures the mean squared difference between predicted probability distribution and one-hot encoded ground truth.

$$ BS = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (f(x_i)_k - y_{i,k})^2 $$

---

## 4. Geospatial Implications of Uncertainty

### 4.1 Domain Shift Detection
High epistemic uncertainty is a robust detector of Out-of-Distribution (OOD) data.
*   *Scenario:* Model trained on European Sentinel-2 data applied to Amazon Rainforest.
*   *Result:* High epistemic uncertainty due to unseen spectral signatures.

### 4.2 Active Learning
Ununennium uses uncertainty heatmaps to guide labeling efforts.
*   **Acquisition Function:** Select patches maximizing entropy $H(y)$.
*   *Benefit:* Reduces labeling costs by 40-60% by focusing on "confusing" geography.

### 4.3 Reliability Diagrams in Mapping
When producing a flood map, we mask out pixels where $H(y) > \tau$ (threshold).
*   **Risk-Averse Policy:** High $\tau$. Only map definite floods.
*   **Risk-Seeking Policy:** Low $\tau$. Map potential floods.

---

## 5. Ununennium Implementation

The `ununennium.metrics.uncertainty` module provides vectorized implementations of these concepts.

```python
# Pseudo-code for Monte Carlo Dropout Evaluation
def estimate_uncertainty(model, image, T=30):
    model.train() # Enable Dropout
    preds = []
    
    for _ in range(T):
        preds.append(torch.softmax(model(image), dim=1))
        
    preds = torch.stack(preds) # (T, C, H, W)
    
    mean_pred = preds.mean(dim=0)
    
    # Entropy (Total Uncertainty)
    entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-6), dim=0)
    
    # Variance (Epistemic Proxy)
    variance = preds.var(dim=0).mean(dim=0)
    
    return mean_pred, entropy, variance
```
