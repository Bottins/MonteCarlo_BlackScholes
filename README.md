# Monte Carlo Black-Scholes (Multi-Dimensional)

## Overview

A research-grade implementation of Monte Carlo methods for pricing and calibrating options under a multi-dimensional Black-Scholes framework. The project is structured around a progressive validation strategy: the simulator is first benchmarked against an analytically exact case (Geometric Basket Call), then extended to cases without closed-form solutions (Arithmetic Basket), and finally applied to an inverse problem — recovering the implied correlation matrix from observed option prices. A dedicated noise sensitivity analysis quantifies calibration robustness under realistic market data conditions.

---

## Key Features

- **Validated Monte Carlo Engine**: Exact convergence verification against a closed-form analytical solution, with empirical confirmation of the theoretical O(1/√N) error decay
- **Geometric Basket Option**: Analytical pricing via the Black-Scholes formula for log-normal geometric means, used as a benchmark throughout the project
- **Arithmetic Basket Option**: Monte Carlo pricing with Control Variate technique (geometric basket as variate), achieving ~99.5% variance reduction
- **Market Data Calibration**: Automatic download and caching of historical prices via `yfinance`; volatilities and correlations estimated from log-returns
- **Inverse Problem — Implied Correlation**: Recovery of the implied equicorrelation parameter ρ from observed basket prices, using Brent root-finding (single strike) and L-BFGS-B least squares (multi-strike)
- **Noise Sensitivity Analysis**: Systematic study of how bid-ask spread and measurement noise propagate into calibration error, comparing single-strike vs multi-strike robustness
- **N-Asset Generic**: All components are parameterized in N; changing the `TICKERS` list is sufficient to run on any number of assets

---

## Scientific Background

### 1. Risk-Neutral Pricing Framework

Under the risk-neutral measure $\mathbb{Q}$, all discounted asset prices are martingales. Option prices are computed as expected discounted payoffs:

$$C_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\left[\max\left(A_T - K,\; 0\right)\right]$$

where $r$ is the continuously compounded risk-free rate, $T$ the maturity, $K$ the strike, and $A_T$ the underlying at expiry.

---

### 2. Multi-Dimensional Geometric Brownian Motion

Under $\mathbb{Q}$, $N$ correlated assets follow the **multi-dimensional Black-Scholes SDE**:

$$dS_i = r\, S_i\, dt + \sigma_i\, S_i\, dW_i^{\mathbb{Q}}, \qquad i = 1, \ldots, N$$

**Brownian correlation structure:**

$$dW_i\, dW_j = \rho_{ij}\, dt, \qquad R = (\rho_{ij})_{i,j} \in \mathbb{R}^{N \times N}, \quad R \succ 0$$

**Model assumptions (GBM):** log-returns are normally distributed and i.i.d. across non-overlapping intervals; volatilities $\sigma_i$ and correlations $\rho_{ij}$ are constant; markets are complete, frictionless, and allow continuous trading.

---

### 3. Exact Numerical Scheme — Itô's Lemma

Applying Itô's formula to $f(S_i) = \ln S_i$ yields an **exact closed-form solution** — the SDE is integrated analytically, with zero time-stepping error:

$$\ln S_i(T) = \ln S_i(0) + \left(r - \frac{\sigma_i^2}{2}\right)T + \sigma_i \sqrt{T}\, Z_i$$

The term $-\sigma_i^2/2$ is the **Itô correction**, arising from the quadratic variation $d\langle \ln S_i \rangle = \sigma_i^2\, dt$. Without it the simulation would be biased upward.

**Cholesky decomposition** generates correlated Gaussian increments from i.i.d. samples:

$$R = L L^\top, \qquad \mathbf{Z} = L\,\mathbf{Z}_{iid}, \qquad \mathbf{Z}_{iid} \sim \mathcal{N}(\mathbf{0},\, I_N)$$

For two assets this gives explicitly:

$$L = \begin{pmatrix} \sigma_1 & 0 \\ \rho\,\sigma_2 & \sigma_2\sqrt{1-\rho^2} \end{pmatrix}$$

---

### 4. Monte Carlo Estimator and Convergence

Given $N$ i.i.d. terminal simulations $\{S^{(k)}(T)\}_{k=1}^N$, the **Monte Carlo price** is:

$$\hat{C}_{MC} = e^{-rT}\,\frac{1}{N}\sum_{k=1}^{N} \max\left(A_T^{(k)} - K,\; 0\right)$$

By the Central Limit Theorem, the standard error decays as:

$$\mathrm{SE}(\hat{C}_{MC}) = \frac{\hat{\sigma}_{\text{payoff}}}{\sqrt{N}} = \mathcal{O}\left(\frac{1}{\sqrt{N}}\right)$$

This is the fundamental convergence rate of MC — dimension-independent, but slow. Doubling accuracy requires $4\times$ more paths.

---

### 5. Geometric Basket — Closed-Form Reference

The geometric mean $G_T = \left(\prod_{i=1}^N S_i(T)\right)^{1/N}$ is itself log-normal:

$$\ln G_T \sim \mathcal{N}(m,\, v^2)$$

$$m = \frac{1}{N}\sum_{i=1}^{N}\left[\ln S_i(0) + \left(r - \frac{\sigma_i^2}{2}\right)T\right], \qquad v^2 = \frac{T}{N^2}\,\boldsymbol{\sigma}^\top R\,\boldsymbol{\sigma}$$

This yields an **exact Black-Scholes formula** (Gentle, 1993):

$$C_{\text{geo}} = e^{-rT}\left[e^{m + v^2/2}\,\Phi(d_1) - K\,\Phi(d_2)\right]$$

$$d_{1,2} = \frac{m - \ln K \pm v^2}{v}$$

This solution serves as **ground truth** throughout: the MC engine is validated by verifying that $\hat{C}_{MC}^{\text{geo}} \to C_{\text{geo}}$ at rate $\mathcal{O}(1/\sqrt{N})$.

---

### 6. Arithmetic Basket and Control Variate

The arithmetic mean $A_T = \frac{1}{N}\sum_i S_i(T)$ is a weighted sum of log-normals — **no closed form exists**. The **Control Variate** technique exploits the high correlation between arithmetic and geometric payoffs on the *same* sample paths:

$$\hat{C}_{CV} = \hat{C}_{MC}^A + c^*\left(C_{\text{geo}}^{\text{analytical}} - \hat{C}_{MC}^G\right)$$

$$c^* = \frac{\mathrm{Cov}\left(\mathrm{payoff}_A,\, \mathrm{payoff}_G\right)}{\mathrm{Var}\left(\mathrm{payoff}_G\right)}$$

The variance reduction factor is $1 - \mathrm{Corr}^2(A_{\text{payoff}},\, G_{\text{payoff}})$, empirically **95–99.5%** for equity baskets.

---

### 7. Implied Correlation — Inverse Problem

The forward problem maps a correlation structure $\rho$ to an option price $C(\rho)$. Under the **equicorrelation model** ($\rho_{ij} = \rho$ for $i \ne j$, $\rho_{ii} = 1$), the inverse problem reduces to a scalar equation:

$$C_{\text{market}} = C(\rho) \quad \Longrightarrow \quad \text{find } \rho \in \left(-\tfrac{1}{N-1},\; 1\right)$$

Solved via **Brent's method** (single strike) or **L-BFGS-B least squares** across a strike grid $\{K_k\}$:

$$\hat{\rho} = \arg\min_{\rho}\, \sum_{k} \left(C_{\text{market}}(K_k) - C(\rho;\, K_k)\right)^2$$

The multi-strike formulation is over-determined and significantly more robust to noise.

---

### 8. Noise Sensitivity

Observed option prices carry bid-ask spread noise. The perturbation model is:

$$C_k^{\text{noisy}} = C_k^{\text{true}} \cdot \left(1 + \sigma_{\text{noise}}\, \varepsilon_k\right), \qquad \varepsilon_k \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$$

By varying $\sigma_{\text{noise}}$ and repeating the inversion over many realisations, we characterise bias, standard deviation, and RMSE of the recovered $\hat{\rho}$ as a function of noise level.

---

## Project Structure

```
MonteCarlo_BlackScholes/
├── mc_black_scholes.py       # Script 1: MC validation vs analytical solution
│                             #   - Market data download and calibration
│                             #   - Geometric basket analytical pricing
│                             #   - Monte Carlo convergence analysis
│                             #   - Convergence plots (3 panels)
│
├── mc_extensions.py          # Script 2: Extensions and inverse problem
│                             #   [A] Arithmetic basket — MC grezzo vs CV
│                             #   [B] Implied correlation (inverse problem)
│                             #   [C] Noise sensitivity analysis
│
├── market_data_cache.pkl     # Cached Yahoo Finance data (auto-generated)
├── convergence_plot.png      # Output: MC convergence vs analytical price
├── extensions_plot.png       # Output: arithmetic basket + forward model
└── noise_sensitivity_plot.png # Output: calibration robustness under noise
```

---

## Technical Stack

- **Numerical Core**: NumPy, SciPy (`linalg.cholesky`, `optimize.brentq`, `optimize.minimize`)
- **Statistical Distributions**: SciPy (`stats.norm`)
- **Market Data**: yfinance (auto-installed if absent)
- **Visualization**: Matplotlib
- **Data Handling**: Pandas

---

## Installation

```bash
pip install numpy scipy pandas matplotlib yfinance
```

Python ≥ 3.10 recommended (uses `X | Y` type union hints).

---

## Usage

### Script 1 — Validation

```bash
python mc_black_scholes.py
```

Downloads historical data (or uses synthetic fallback), calibrates σ and ρ, prices the Geometric Basket Call analytically and via Monte Carlo at multiple path counts, and produces convergence statistics and plots.

**Key configuration** (top of file):

| Parameter       | Default                               | Description                              |
|-----------------|---------------------------------------|------------------------------------------|
| `TICKERS`       | `["AAPL","MSFT","GOOGL","AMZN","META"]` | Assets (any length N)                   |
| `T`             | `1.0`                                 | Maturity in years                        |
| `r`             | `0.05`                                | Risk-free rate                           |
| `K_MONEYNESS`   | `1.0`                                 | Strike = moneyness × G₀ (1.0 = ATM)     |
| `MC_SIZES`      | `[500, …, 500_000]`                   | Path counts for convergence analysis     |
| `N_REPEATS`     | `30`                                  | Independent runs per path count          |

### Script 2 — Extensions and Inverse Problem

```bash
python mc_extensions.py
```

Runs all three sections sequentially. Reads the same market cache produced by Script 1; no additional download required.

**Key configuration** (top of file):

| Parameter        | Default                        | Description                                  |
|------------------|--------------------------------|----------------------------------------------|
| `MC_SIZES_EXT`   | `[1_000, …, 200_000]`          | Path counts for arithmetic basket analysis   |
| `MONEYNESS_GRID` | `[0.85, 0.90, …, 1.15]`        | Strike grid for implied correlation          |
| `N_PATHS_INVERSE`| `100_000`                      | Paths for MC forward model in calibration    |
| `NOISE_LEVELS`   | `[0.0, 0.001, …, 0.10]`        | Relative noise levels σ_noise                |
| `N_NOISE_REPS`   | `300`                          | Calibration repeats per noise level          |

---

## Results Summary

### Convergence (Script 1)

| N paths   | MC price  | Std MC   | Error rel. |
|-----------|-----------|----------|------------|
| 500       | converges | ~2.4     | ~0.4%      |
| 10,000    | converges | ~0.49    | <0.5%      |
| 500,000   | converges | ~0.075   | ~0.025%    |

Std decreases at the theoretical rate: Std(N=500k) / Std(N=500) ≈ √(500/500,000) ✓

### Variance Reduction — Control Variate (Script 2)

| N paths   | Std grezzo | Std CV   | Variance reduction |
|-----------|------------|----------|--------------------|
| 1,000     | 0.510      | 0.035    | ~99.5%             |
| 200,000   | 0.044      | 0.003    | ~99.7%             |

### Noise Sensitivity — Implied Correlation (Script 2)

| σ_noise | RMSE (singolo K) | RMSE (multi-K) | Improvement |
|---------|------------------|----------------|-------------|
| 0.1%    | 0.0015           | 0.0006         | 2.4×        |
| 1%      | 0.0156           | 0.0067         | 2.3×        |
| 5%      | 0.0730           | 0.0362         | 2.0×        |
| 10%     | 0.150            | 0.068          | 2.2×        |

The multi-strike calibration consistently halves the RMSE relative to single-strike across all noise levels. Bias is negligible at all levels, confirming that the inverse problem is well-posed under the equicorrelation assumption.

---

## References

- Gentle, D. (1993). *Basket Weaving*. Risk Magazine, 6(6), 51–52.
- Carmona, R. & Durrleman, V. (2003). *Pricing and hedging spread options*. SIAM Review, 45(4), 627–685.
- Broadie, M. & Glasserman, P. (1996). *Estimating security price derivatives using simulation*. Management Science, 42(2), 269–285.
- Kemna, A. & Vorst, A. (1990). *A pricing method for options based on average asset values*. Journal of Banking & Finance, 14(1), 113–129.
- Black, F. & Scholes, M. (1973). *The pricing of options and corporate liabilities*. Journal of Political Economy, 81(3), 637–654.
