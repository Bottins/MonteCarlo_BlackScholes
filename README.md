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

### Multi-Dimensional Black-Scholes

Under the risk-neutral measure, N correlated assets follow Geometric Brownian Motion:

```
dSᵢ = r Sᵢ dt + σᵢ Sᵢ dWᵢ,    dWᵢ dWⱼ = ρᵢⱼ dt
```

The terminal log-prices are simulated exactly (no time-stepping error) via Cholesky decomposition of the correlation matrix R = L Lᵀ:

```
log Sᵢ(T) = log Sᵢ(0) + (r − σᵢ²/2) T + σᵢ √T Zᵢ
Z = L Z_iid,    Z_iid ~ N(0, I)
```

### Geometric Basket — Closed-Form Reference

The geometric mean G_T = (∏ᵢ Sᵢ(T))^(1/N) is itself log-normal:

```
log G_T ~ N(m, v²)

m  = (1/N) Σᵢ [ log Sᵢ(0) + (r − σᵢ²/2) T ]
v² = (T / N²) σᵀ R σ
```

This admits an exact Black-Scholes formula (Gentle, 1993):

```
C_geo = e^{−rT} [ e^{m + v²/2} Φ(d₁) − K Φ(d₂) ]
d₁ = (m − ln K + v²) / v,    d₂ = d₁ − v
```

This solution is used throughout as **ground truth** for validating the Monte Carlo engine.

### Arithmetic Basket — No Closed Form

The arithmetic mean A_T = (1/N) Σᵢ Sᵢ(T) is a sum of log-normals, which has no closed-form distribution. Monte Carlo is the standard numerical approach. The **Control Variate** technique exploits the high correlation between geometric and arithmetic payoffs on the same sample:

```
Â_cv = Â_MC + c* · (C_geo_analytical − Ĝ_MC)
c*   = Cov(A_payoff, G_payoff) / Var(G_payoff)
```

This reduces estimator variance by a factor of 1 − Corr²(A, G), typically 95–99%.

### Implied Correlation — Inverse Problem

The forward problem maps a correlation structure to an option price. Under the **equicorrelation model** (R_ij = ρ for i≠j, R_ii = 1), the inverse problem reduces to a scalar equation:

```
C_market = C(ρ)    →    find ρ ∈ (−1/(N−1), 1)
```

Solved via Brent's method on a single strike, or L-BFGS-B least squares across a strike grid. The multi-strike formulation is over-determined and significantly more robust to noise.

### Noise Sensitivity

In practice, observed option prices carry bid-ask noise. The noise model used is:

```
C_noisy_k = C_true_k · (1 + σ_noise · εₖ),    εₖ ~ N(0, 1)
```

By varying σ_noise and repeating the calibration over many realizations, we obtain the distribution of the recovered ρ and characterize bias, standard deviation, and RMSE as functions of noise level.

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
