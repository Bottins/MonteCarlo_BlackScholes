#!/usr/bin/env python3
"""
================================================================================
  Estensioni MC Black-Scholes Multidimensionale
  1) Arithmetic Basket Option  (nessuna soluzione analitica)
  2) Problema Inverso: Implied Correlation (calibrazione di ρ)
================================================================================

LOGICA DEL PROGETTO
-------------------
Lo script mc_black_scholes.py ha validato il simulatore Monte Carlo su un caso
con soluzione esatta nota (Geometric Basket). Questo script estende il lavoro
in due direzioni:

  [A]  Arithmetic Basket  —  nessuna formula chiusa
       ┌─────────────────────────────────────────────────────────┐
       │  Payoff: max( (1/N) Σ Sᵢ(T) − K, 0 )                 │
       │  La somma di log-normali NON è log-normale              │
       │  → non esiste soluzione B-S analitica                   │
       └─────────────────────────────────────────────────────────┘
       Approcci usati:
       • MC grezzo        : stima diretta del payoff
       • Control Variate  : corregge MC aritmetico col geometrico
                            (di cui si conosce il prezzo esatto)
                            → riduzione di varianza teoricamente ottimale
       Il prezzo geometrico funge da lower bound (AM-GM) e da
       riferimento di plausibilità.

  [B]  Implied Correlation  —  problema inverso
       ┌─────────────────────────────────────────────────────────┐
       │  Forward:  dato ρ  →  C_basket(ρ)  via MC              │
       │  Inverso:  dato C_obs  →  trova ρ  t.c. C(ρ) ≈ C_obs  │
       └─────────────────────────────────────────────────────────┘
       Modello equicorrelazione: R_ij = ρ per i≠j, R_ii = 1
       → 1 solo parametro → root-finding 1D (Brent, stabile)

       Test sintetico: genero prezzi "di mercato" con ρ_vero noto,
       poi inverto e verifico che il recupero funzioni.
       Esteso a più strike contemporaneamente → sistema sovra-determinato
       → calibrazione ai minimi quadrati.

RIFERIMENTI
-----------
  - Kemna & Vorst (1990). "A pricing method for options based on average
    asset values." Journal of Banking & Finance.
  - Broadie & Glasserman (1996). "Estimating security price derivatives
    using simulation." Management Science.
  - Cont & Tankov (2004). Financial Modelling with Jump Processes. CRC Press.
================================================================================
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky
from scipy.optimize import brentq, minimize

# Importa le funzioni di base dal primo script
sys.path.insert(0, str(Path(__file__).parent))
from mc_black_scholes import (
    download_or_load,
    calibrate,
    synthetic_params,
    ensure_psd,
    geometric_basket_analytical,
    geometric_basket_mc,
    TICKERS, T, r, K_MONEYNESS, LOOKBACK_DAYS, CACHE_FILE, SEED,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE ESTENSIONI
# ─────────────────────────────────────────────────────────────────────────────

MC_SIZES_EXT  = [1_000, 5_000, 10_000, 50_000, 200_000]
N_REPEATS_EXT = 30

# Per il problema inverso: griglia di strike (moneyness rispetto a G0)
MONEYNESS_GRID = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

# Numero di path MC per il forward model nell'inversione
# (più alto = meno rumore, ma più lento)
N_PATHS_INVERSE = 100_000


# ─────────────────────────────────────────────────────────────────────────────
# PARTE A — ARITHMETIC BASKET
# ─────────────────────────────────────────────────────────────────────────────

def arithmetic_basket_mc_raw(S0, sigma, corr, K, r, T, n_paths, rng):
    """
    MC grezzo per la call su paniere aritmetico.

    Payoff: max( (1/N) Σ Sᵢ(T) − K , 0 )

    La somma di log-normali non è log-normale → nessuna formula chiusa.
    Simulazione identica al caso geometrico, cambia solo l'aggregazione finale.

    Returns
    -------
    price : float  –  stima MC scontata
    se    : float  –  errore standard
    """
    N = len(S0)
    L = cholesky(corr, lower=True)

    Z    = rng.standard_normal((N, n_paths))
    W    = L @ Z

    log_ST = (
        np.log(S0) + (r - 0.5 * sigma**2) * T
    )[:, None] + sigma[:, None] * np.sqrt(T) * W    # (N, n_paths)

    ST         = np.exp(log_ST)                      # prezzi terminali
    arith_mean = ST.mean(axis=0)                     # (n_paths,)  media aritmetica
    payoff     = np.maximum(arith_mean - K, 0.0)

    disc  = np.exp(-r * T)
    price = disc * payoff.mean()
    se    = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return float(price), float(se)


def arithmetic_basket_mc_cv(S0, sigma, corr, K, r, T, n_paths, rng):
    """
    MC con Control Variate geometrico per la call su paniere aritmetico.

    L'idea: il paniere geometrico G e quello aritmetico A sono altamente
    correlati sullo stesso campione. Poiché il prezzo esatto di G è noto,
    possiamo usarlo per correggere la stima di A:

        Â_cv = Â_MC + c* · (C_geo_analitico − Ĝ_MC)

    Il coefficiente ottimale è:
        c* = Cov(A_payoff, G_payoff) / Var(G_payoff)

    stimato sullo stesso campione MC. Questo riduce la varianza di un fattore

        1 − Corr²(A_payoff, G_payoff)

    tipicamente ~60–90% in meno.

    Returns
    -------
    price_cv : float  –  stima corretta col control variate
    se_cv    : float  –  errore standard ridotto
    price_raw: float  –  stima grezza (per confronto)
    """
    N = len(S0)
    L = cholesky(corr, lower=True)

    Z    = rng.standard_normal((N, n_paths))
    W    = L @ Z

    log_ST = (
        np.log(S0) + (r - 0.5 * sigma**2) * T
    )[:, None] + sigma[:, None] * np.sqrt(T) * W

    ST         = np.exp(log_ST)

    # Payoff aritmetico
    arith_mean = ST.mean(axis=0)
    pa         = np.maximum(arith_mean - K, 0.0)   # (n_paths,)

    # Payoff geometrico (stessa simulazione)
    log_G  = log_ST.mean(axis=0)
    pg     = np.maximum(np.exp(log_G) - K, 0.0)    # (n_paths,)

    # Prezzo esatto geometrico (la nostra "verità" per il control variate)
    C_geo_exact = geometric_basket_analytical(S0, sigma, corr, K, r, T)

    # Coefficiente ottimale c* stimato dal campione
    # Formula: c* = Cov(Y, X) / Var(X)  con Y=payoff aritm., X=payoff geom.
    cov_matrix = np.cov(pa, pg, ddof=1)
    c_star     = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-14)

    # E[pg] (non scontato) = C_geo_exact / disc
    mu_pg = C_geo_exact / np.exp(-r * T)

    # Correzione: Y_cv = Y + c* * (μ_X - X)  →  segno MENO rispetto a (X - μ_X)
    # Quando pg > μ_pg  (geo alto → arith alto → sovrastimiamo) correggiamo in giù
    disc       = np.exp(-r * T)
    pa_cv      = pa + c_star * (mu_pg - pg)

    price_cv  = disc * pa_cv.mean()
    se_cv     = disc * pa_cv.std(ddof=1) / np.sqrt(n_paths)
    price_raw = disc * pa.mean()

    return float(price_cv), float(se_cv), float(price_raw)


def analyze_arithmetic_basket(S0, sigma, corr, K, r, T, mc_sizes, n_repeats, seed):
    """
    Confronto MC grezzo vs Control Variate per il paniere aritmetico.
    Poiché non esiste soluzione esatta, usiamo il valore MC con N=200k
    ripetuto molte volte come "stima di riferimento".
    """
    # Stima di riferimento (MC con molti path, usata come proxy per la verità)
    print("  Calcolo stima di riferimento (N=1M, 5 run) ...")
    ref_prices = []
    for i in range(5):
        rng = np.random.default_rng(seed + i * 3333)
        p, _ = arithmetic_basket_mc_raw(S0, sigma, corr, K, r, T, 1_000_000, rng)
        ref_prices.append(p)
    ref_price = np.mean(ref_prices)
    ref_std   = np.std(ref_prices)
    print(f"  Stima riferimento:  {ref_price:.6f}  ±{1.96*ref_std/np.sqrt(5):.6f}  (IC 95%)")
    print(f"  [nota] Non è un valore esatto, è la miglior stima MC disponibile.")

    geo_price = geometric_basket_analytical(S0, sigma, corr, K, r, T)
    print(f"  Prezzo geometrico (lower bound analitico): {geo_price:.6f}")
    print(f"  Differenza arith−geo: {ref_price - geo_price:.6f}  "
          f"({(ref_price - geo_price) / geo_price:.3%})")

    print(f"\n  {'N paths':>10}  {'MC grezzo':>12}  {'Std grezzo':>11}  "
          f"{'MC + CV':>12}  {'Std CV':>11}  {'Rid. varianza':>14}")
    print("  " + "─" * 80)

    records = []
    for n in mc_sizes:
        raw_prices, cv_prices, se_raws, se_cvs = [], [], [], []
        for rep in range(n_repeats):
            rng = np.random.default_rng(seed + rep * 1009 + n % 100_000)
            p_cv, se_cv, p_raw = arithmetic_basket_mc_cv(S0, sigma, corr, K, r, T, n, rng)
            rng2 = np.random.default_rng(seed + rep * 1009 + n % 100_000)
            _, se_raw = arithmetic_basket_mc_raw(S0, sigma, corr, K, r, T, n, rng2)
            raw_prices.append(p_raw)
            cv_prices.append(p_cv)
            se_raws.append(se_raw)
            se_cvs.append(se_cv)

        std_raw  = np.std(raw_prices, ddof=1)
        std_cv   = np.std(cv_prices, ddof=1)
        var_red  = 1 - (std_cv / std_raw) ** 2   # riduzione di varianza empirica

        print(f"  {n:>10,}  {np.mean(raw_prices):>12.6f}  {std_raw:>11.6f}  "
              f"{np.mean(cv_prices):>12.6f}  {std_cv:>11.6f}  "
              f"{var_red:>13.1%}")

        records.append({
            "n_paths": n,
            "mean_raw": np.mean(raw_prices), "std_raw": std_raw,
            "mean_cv":  np.mean(cv_prices),  "std_cv":  std_cv,
            "var_reduction": var_red,
        })

    return ref_price, geo_price, records


# ─────────────────────────────────────────────────────────────────────────────
# PARTE B — IMPLIED CORRELATION (PROBLEMA INVERSO)
# ─────────────────────────────────────────────────────────────────────────────

def equicorrelation_matrix(rho: float, n: int) -> np.ndarray:
    """
    Costruisce la matrice di equicorrelazione:
        R_ij = rho  per i≠j,  R_ii = 1

    È definita positiva per  rho > -1/(n-1).
    """
    R = np.full((n, n), rho)
    np.fill_diagonal(R, 1.0)
    return R


def price_given_rho(rho: float, S0, sigma, K, r, T, n_paths, seed,
                    use_geo: bool = False) -> float:
    """
    Funzione forward: dato ρ → C_basket.

    Se use_geo=True usa la formula analitica del geometrico (più veloce,
    per visualizzazione della curva C(ρ)).
    Se use_geo=False usa MC sull'aritmetico (più realistico per l'inversione).
    """
    N    = len(S0)
    corr = equicorrelation_matrix(rho, N)

    # Controllo PSD
    if np.linalg.eigvalsh(corr).min() < 1e-8:
        return np.nan

    if use_geo:
        return geometric_basket_analytical(S0, sigma, corr, K, r, T)
    else:
        rng = np.random.default_rng(seed)
        p, _ = arithmetic_basket_mc_raw(S0, sigma, corr, K, r, T, n_paths, rng)
        return p


def implied_correlation_single(C_obs: float, S0, sigma, K, r, T,
                                 n_paths: int, seed: int,
                                 rho_bounds: tuple = None,
                                 use_geo: bool = False) -> float:
    """
    Recupera ρ implicita da un singolo prezzo osservato C_obs.

    Usa il metodo di Brent (root-finding su intervallo chiuso):
        f(ρ) = C(ρ) − C_obs = 0

    Il bound inferiore è -1/(N-1)+ε, cioè il minimo valore per cui la
    matrice di equicorrelazione N×N è definita positiva.

    Returns ρ_implied, o np.nan se fuori range / non convergente.
    """
    N = len(S0)
    rho_min = -1.0 / (N - 1) + 1e-3   # limite PSD per equicorrelazione N×N
    if rho_bounds is None:
        rho_bounds = (rho_min, 0.999)

    def objective(rho):
        c = price_given_rho(rho, S0, sigma, K, r, T, n_paths, seed, use_geo)
        return c - C_obs

    # Verifica che C_obs sia all'interno del range della curva C(ρ)
    c_lo = price_given_rho(rho_bounds[0], S0, sigma, K, r, T, n_paths, seed, use_geo)
    c_hi = price_given_rho(rho_bounds[1], S0, sigma, K, r, T, n_paths, seed, use_geo)

    if np.isnan(c_lo) or np.isnan(c_hi):
        return np.nan
    if not (min(c_lo, c_hi) <= C_obs <= max(c_lo, c_hi)):
        return np.nan

    try:
        rho_impl = brentq(objective, rho_bounds[0], rho_bounds[1], xtol=1e-4, maxiter=100)
        return float(rho_impl)
    except Exception:
        return np.nan


def implied_correlation_lsq(C_obs_list: list, K_list: list, S0, sigma, r, T,
                              n_paths: int, seed: int,
                              use_geo: bool = False) -> dict:
    """
    Calibrazione ai minimi quadrati su più strike simultaneamente.

    Problema: trova ρ che minimizza  Σₖ (C_MC(ρ, Kₖ) − C_obs_k)²

    Con più strike il sistema è sovra-determinato → calibrazione più robusta
    al rumore MC. Usa L-BFGS-B con ρ ∈ (−0.95, 0.999).

    Returns dict con ρ_calibrato, errori, etc.
    """
    def total_loss(rho_arr):
        rho = rho_arr[0]
        loss = 0.0
        for K, C_obs in zip(K_list, C_obs_list):
            C_mod = price_given_rho(rho, S0, sigma, K, r, T, n_paths, seed, use_geo)
            if np.isnan(C_mod):
                return 1e10
            loss += (C_mod - C_obs) ** 2
        return loss

    result = minimize(
        total_loss,
        x0=[0.3],
        bounds=[(-0.95, 0.999)],
        method="L-BFGS-B",
        options={"ftol": 1e-10, "gtol": 1e-6, "maxiter": 200},
    )
    return {"rho": float(result.x[0]), "loss": float(result.fun),
            "success": result.success, "message": result.message}


def analyze_inverse_problem(S0, sigma, rho_true: float, r, T,
                             moneyness_grid: list, n_paths: int, seed: int):
    """
    Test sintetico del problema inverso.

    1. Genera prezzi "di mercato" con ρ_vero noto (usando la formula analitica
       geometrica come proxy stabile, poi replicato anche con MC aritmetico).
    2. Usa questi prezzi come input e recupera ρ_implicita.
    3. Confronta ρ_recuperata con ρ_vera.

    Tutto fatto in modalità use_geo=True per avere un test pulito senza
    rumore MC nel forward model (per la calibrazione vera, use_geo=False).
    """
    N    = len(S0)
    G0   = float(np.exp(np.mean(np.log(S0))))

    K_list     = [m * G0 for m in moneyness_grid]
    corr_true  = equicorrelation_matrix(rho_true, N)
    corr_true  = ensure_psd(corr_true)

    # Genera prezzi "di mercato" con formula analitica (ground truth pulito)
    C_market = [
        geometric_basket_analytical(S0, sigma, corr_true, K, r, T)
        for K in K_list
    ]

    print(f"\n  ρ_vero usato per generare i prezzi: {rho_true:.3f}")
    print(f"\n  {'Moneyness':>11}  {'K':>10}  {'C_market':>12}")
    print("  " + "─" * 38)
    for m, K, C in zip(moneyness_grid, K_list, C_market):
        print(f"  {m:>11.2f}  {K:>10.4f}  {C:>12.6f}")

    # ── Inversione su singolo strike (ATM) ─────────────────────────────────
    idx_atm   = moneyness_grid.index(1.0) if 1.0 in moneyness_grid else len(moneyness_grid) // 2
    C_atm     = C_market[idx_atm]
    K_atm     = K_list[idx_atm]
    rho_single = implied_correlation_single(
        C_atm, S0, sigma, K_atm, r, T, n_paths, seed, use_geo=True
    )

    print(f"\n  Inversione singolo strike (ATM, moneyness=1.0):")
    print(f"    C_obs   = {C_atm:.6f}")
    print(f"    ρ_vera  = {rho_true:.4f}")
    print(f"    ρ_imp.  = {rho_single:.4f}  (errore: {abs(rho_single - rho_true):.4f})")

    # ── Calibrazione multi-strike ────────────────────────────────────────────
    res_lsq = implied_correlation_lsq(
        C_market, K_list, S0, sigma, r, T, n_paths, seed, use_geo=True
    )
    print(f"\n  Calibrazione multi-strike ({len(K_list)} strike, minimi quadrati):")
    print(f"    ρ_vera      = {rho_true:.4f}")
    print(f"    ρ_calibrata = {res_lsq['rho']:.4f}  "
          f"(errore: {abs(res_lsq['rho'] - rho_true):.4f})")
    print(f"    Loss finale = {res_lsq['loss']:.4e}   success={res_lsq['success']}")

    return K_list, C_market, rho_single, res_lsq


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZZAZIONE
# ─────────────────────────────────────────────────────────────────────────────

def plot_extensions(geo_price, ref_arith, records_arith,
                    K_list, C_market, rho_true, rho_single, res_lsq,
                    S0, sigma, r, T, moneyness_grid,
                    tickers, output_dir: Path):

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Estensioni MC Black-Scholes  —  Asset: {', '.join(tickers)}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Pannello A: riduzione di varianza Control Variate ──────────────────
    ax = axes[0]
    ns      = [rec["n_paths"]       for rec in records_arith]
    std_raw = [rec["std_raw"]       for rec in records_arith]
    std_cv  = [rec["std_cv"]        for rec in records_arith]
    var_red = [rec["var_reduction"] for rec in records_arith]

    ax.loglog(ns, std_raw, "o--", color="steelblue",  lw=2, ms=6, label="MC grezzo")
    ax.loglog(ns, std_cv,  "s-",  color="darkorange", lw=2, ms=6, label="MC + Control Variate")
    ax.set_xlabel("N simulazioni (log)", fontsize=11)
    ax.set_ylabel("Std della stima (log)", fontsize=11)
    ax.set_title("(A)  Arithmetic Basket\nRiduzione di varianza — MC vs Control Variate", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.35)

    # Annotazione riduzione media
    avg_red = np.mean(var_red)
    ax.annotate(
        f"Rid. var. media: {avg_red:.1%}",
        xy=(0.05, 0.12), xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )

    # ── Pannello B: curva prezzo vs ρ (forward model) ──────────────────────
    ax = axes[1]
    N    = len(S0)
    G0   = float(np.exp(np.mean(np.log(S0))))
    K_atm = 1.0 * G0

    N_plot    = len(S0)
    rho_min_plot = -1.0 / (N_plot - 1) + 0.01
    rho_grid  = np.linspace(rho_min_plot, 0.95, 60)
    C_geo_rho = []
    for rho in rho_grid:
        corr_r = equicorrelation_matrix(rho, N)
        if np.linalg.eigvalsh(corr_r).min() < 1e-8:
            C_geo_rho.append(np.nan)
        else:
            C_geo_rho.append(geometric_basket_analytical(S0, sigma, corr_r, K_atm, r, T))

    ax.plot(rho_grid, C_geo_rho, "-", color="mediumseagreen", lw=2.5,
            label="C_geo(ρ)  [analitico, ATM]")
    ax.axvline(rho_true, color="crimson", lw=2, ls="--", label=f"ρ_vero = {rho_true:.2f}")
    ax.axvline(rho_single, color="navy", lw=1.5, ls=":", label=f"ρ_impl. = {rho_single:.3f}")
    ax.axhline(
        geometric_basket_analytical(
            S0, sigma, equicorrelation_matrix(rho_true, N), K_atm, r, T
        ),
        color="gray", lw=1, ls="-.",
        label=f"C_obs (ATM) = {C_market[len(C_market)//2]:.4f}",
    )
    ax.set_xlabel("Correlazione ρ (equicorrelazione)", fontsize=11)
    ax.set_ylabel("Prezzo opzione ATM", fontsize=11)
    ax.set_title("(B)  Problema Inverso\nCurva prezzo vs ρ (forward model)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    # ── Pannello C: calibrazione multi-strike ─────────────────────────────
    ax = axes[2]
    N    = len(S0)

    # Prezzo sintetico col ρ_calibrato
    corr_cal  = equicorrelation_matrix(res_lsq["rho"], N)
    C_fitted  = [
        geometric_basket_analytical(S0, sigma, corr_cal, K, r, T)
        for K in K_list
    ]
    C_true_plot = C_market

    ax.plot(moneyness_grid, C_true_plot, "o-", color="crimson", lw=2, ms=7,
            label=f"C_market  (ρ_vero={rho_true:.2f})")
    ax.plot(moneyness_grid, C_fitted,    "s--", color="navy",   lw=2, ms=7,
            label=f"C_fitted  (ρ_cal={res_lsq['rho']:.3f})")
    ax.set_xlabel("Moneyness K/G₀", fontsize=11)
    ax.set_ylabel("Prezzo opzione", fontsize=11)
    ax.set_title(
        "(C)  Calibrazione multi-strike\n"
        f"ρ_vero={rho_true:.3f}  →  ρ_cal={res_lsq['rho']:.3f}  "
        f"(err={abs(res_lsq['rho']-rho_true):.4f})",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    out = output_dir / "extensions_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot salvato in '{out}'")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PARTE C — ANALISI DI SENSITIVITÀ AL RUMORE
# ─────────────────────────────────────────────────────────────────────────────

# Livelli di rumore relativo da esplorare (es. 0.01 = 1% del prezzo)
NOISE_LEVELS  = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.10]
N_NOISE_REPS  = 300   # ripetizioni per ogni livello di rumore


def noise_sensitivity_analysis(S0, sigma, rho_true: float, r, T,
                                moneyness_grid: list,
                                n_paths: int, seed: int) -> list:
    """
    Studia come il rumore sui prezzi osservati si propaga sull'errore
    di calibrazione di ρ.

    Modello di rumore:
        C_noisy_k = C_market_k * (1 + σ_noise * ε_k),   ε_k ~ N(0,1)

    dove σ_noise è il livello di rumore relativo.

    Per ogni σ_noise esegue N_NOISE_REPS estrazioni del vettore di prezzi
    rumorosi e calibra ρ sia su singolo strike (ATM) che su tutti gli strike
    contemporaneamente (multi-K). Raccoglie la distribuzione delle ρ
    recuperate e calcola: bias, std, RMSE.

    Returns
    -------
    records : list of dict, uno per ogni livello di rumore
    """
    N    = len(S0)
    G0   = float(np.exp(np.mean(np.log(S0))))
    K_list = [m * G0 for m in moneyness_grid]
    idx_atm = len(moneyness_grid) // 2   # indice strike ATM

    # Prezzi "di mercato" senza rumore (ground truth)
    corr_true  = equicorrelation_matrix(rho_true, N)
    corr_true  = ensure_psd(corr_true)
    C_market   = [geometric_basket_analytical(S0, sigma, corr_true, K, r, T)
                  for K in K_list]

    print(f"\n  ρ_vero = {rho_true:.4f}")
    print(f"  Livelli di rumore relativo: {NOISE_LEVELS}")
    print(f"  Ripetizioni per livello: {N_NOISE_REPS}")
    print(f"\n  {'σ_noise':>9}  {'Bias_1K':>9}  {'Std_1K':>9}  {'RMSE_1K':>9}  "
          f"{'Bias_NK':>9}  {'Std_NK':>9}  {'RMSE_NK':>9}")
    print("  " + "─" * 72)

    records = []
    rng_noise = np.random.default_rng(seed + 777)

    for sigma_noise in NOISE_LEVELS:

        rho_single_list = []
        rho_multi_list  = []

        for _ in range(N_NOISE_REPS):
            # Aggiungi rumore relativo indipendente su ogni strike
            eps      = rng_noise.standard_normal(len(K_list))
            C_noisy  = [C * (1.0 + sigma_noise * e)
                        for C, e in zip(C_market, eps)]
            # Tronca a zero (prezzi non possono essere negativi)
            C_noisy  = [max(c, 1e-6) for c in C_noisy]

            # Calibrazione su singolo strike (ATM)
            rho_s = implied_correlation_single(
                C_noisy[idx_atm], S0, sigma, K_list[idx_atm],
                r, T, n_paths, seed, use_geo=True
            )
            rho_single_list.append(rho_s if not np.isnan(rho_s) else np.nan)

            # Calibrazione multi-strike
            res = implied_correlation_lsq(
                C_noisy, K_list, S0, sigma, r, T, n_paths, seed, use_geo=True
            )
            rho_multi_list.append(res["rho"])

        # Filtra i nan dalla calibrazione singola
        rho_s_arr = np.array([x for x in rho_single_list if not np.isnan(x)])
        rho_m_arr = np.array(rho_multi_list)

        def stats(arr):
            bias = arr.mean() - rho_true
            std  = arr.std(ddof=1)
            rmse = np.sqrt(bias**2 + std**2)
            return bias, std, rmse

        b1, s1, r1 = stats(rho_s_arr)
        bN, sN, rN = stats(rho_m_arr)

        print(f"  {sigma_noise:>9.3f}  {b1:>+9.5f}  {s1:>9.5f}  {r1:>9.5f}  "
              f"{bN:>+9.5f}  {sN:>9.5f}  {rN:>9.5f}")

        records.append({
            "sigma_noise":      sigma_noise,
            "rho_single":       rho_s_arr,
            "rho_multi":        rho_m_arr,
            "bias_single":      b1,  "std_single":  s1,  "rmse_single": r1,
            "bias_multi":       bN,  "std_multi":   sN,  "rmse_multi":  rN,
        })

    return records


def plot_noise_sensitivity(records: list, rho_true: float, output_dir: Path):
    """
    Due pannelli:
      (A) RMSE di ρ_recuperata vs livello di rumore — singolo K vs multi-K
      (B) Boxplot delle distribuzioni di ρ per ogni livello di rumore
          (solo multi-K, più robusto e informativo)
    """
    noise_levels = [rec["sigma_noise"] for rec in records]
    rmse_s = [rec["rmse_single"] for rec in records]
    rmse_m = [rec["rmse_multi"]  for rec in records]
    std_s  = [rec["std_single"]  for rec in records]
    std_m  = [rec["std_multi"]   for rec in records]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Sensitività al Rumore — Problema Inverso (Implied Correlation)\n"
        f"ρ_vero = {rho_true:.3f}   |   {N_NOISE_REPS} realizzazioni per livello",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── Pannello A: RMSE e Std vs σ_noise ────────────────────────────────────
    ax = axes[0]
    nl_pct = [n * 100 for n in noise_levels]

    ax.semilogy(nl_pct, rmse_s, "o-",  color="steelblue",   lw=2, ms=6,
                label="RMSE — singolo strike (ATM)")
    ax.semilogy(nl_pct, rmse_m, "s-",  color="darkorange",  lw=2, ms=6,
                label=f"RMSE — multi-strike ({len(MONEYNESS_GRID)} K)")
    ax.semilogy(nl_pct, std_s,  "o--", color="steelblue",   lw=1.5, ms=4, alpha=0.5,
                label="Std — singolo K")
    ax.semilogy(nl_pct, std_m,  "s--", color="darkorange",  lw=1.5, ms=4, alpha=0.5,
                label="Std — multi-K")

    ax.set_xlabel("Livello di rumore relativo  σ_noise  (%)", fontsize=11)
    ax.set_ylabel("RMSE  /  Std  di  ρ_recuperata  (log)", fontsize=11)
    ax.set_title("(A)  Errore di calibrazione vs rumore", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.35)

    # Annotazione al 5% di rumore
    idx_5pct = next((i for i, n in enumerate(noise_levels) if abs(n - 0.05) < 0.001), -1)
    if idx_5pct >= 0:
        ax.annotate(
            f"@5% rumore:\n  singolo: RMSE={rmse_s[idx_5pct]:.3f}\n"
            f"  multi:   RMSE={rmse_m[idx_5pct]:.3f}",
            xy=(5.0, rmse_s[idx_5pct]),
            xytext=(0.42, 0.75), textcoords="axes fraction",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
        )

    # ── Pannello B: Boxplot distribuzioni ρ ──────────────────────────────────
    ax = axes[1]

    # Usa solo i livelli non-zero per il boxplot (esclude noise=0 che è un punto)
    nonzero = [rec for rec in records if rec["sigma_noise"] > 0]
    labels  = [f"{rec['sigma_noise']*100:.1f}%" for rec in nonzero]

    bp = ax.boxplot(
        [rec["rho_multi"] for rec in nonzero],
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="crimson", linewidth=2),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
        flierprops=dict(marker=".", markersize=3, color="gray", alpha=0.5),
    )
    # Colore degradato in funzione del rumore
    cmap = plt.cm.YlOrRd
    for patch, level in zip(bp["boxes"], nonzero):
        intensity = level["sigma_noise"] / max(NOISE_LEVELS)
        patch.set_facecolor(cmap(0.2 + 0.7 * intensity))
        patch.set_alpha(0.75)

    ax.axhline(rho_true, color="navy", lw=2, ls="--", label=f"ρ_vero = {rho_true:.3f}")
    ax.set_xlabel("Livello di rumore relativo  σ_noise", fontsize=11)
    ax.set_ylabel("ρ_recuperata  (calibrazione multi-K)", fontsize=11)
    ax.set_title("(B)  Distribuzione di ρ_implicita per livello di rumore", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.35)

    plt.tight_layout()
    out = output_dir / "noise_sensitivity_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot salvato in '{out}'")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Estensioni MC — Arithmetic Basket + Implied Correlation")
    print("=" * 70)

    # ── Carica parametri calibrati dallo script principale ─────────────────
    script_dir = Path(__file__).parent
    cache_path = script_dir / CACHE_FILE

    print(f"\n[0] Carico parametri di mercato")
    prices = download_or_load(TICKERS, LOOKBACK_DAYS, str(cache_path))
    if prices is not None:
        S0, sigma, corr_hist = calibrate(prices)
        print(f"  Parametri storici caricati ({TICKERS})")
    else:
        S0, sigma, corr_hist = synthetic_params(len(TICKERS))
        print("  Parametri sintetici (fallback)")

    corr_hist = ensure_psd(corr_hist)

    G0 = float(np.exp(np.mean(np.log(S0))))
    K  = K_MONEYNESS * G0

    # ─────────────────────────────────────────────────────────────────────────
    # PARTE A: Arithmetic Basket
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[A]  ARITHMETIC BASKET  (nessuna soluzione analitica)")
    print("─" * 70)
    print(f"\n  La correlazione storica viene usata come parametro fisso.")
    print(f"  K = {K:.4f}  (ATM),  T = {T}y,  r = {r:.0%}")
    print(f"\n  Avvio analisi MC grezzo vs Control Variate ...")

    ref_arith, geo_price, records_arith = analyze_arithmetic_basket(
        S0, sigma, corr_hist, K, r, T, MC_SIZES_EXT, N_REPEATS_EXT, SEED
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PARTE B: Implied Correlation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[B]  PROBLEMA INVERSO — IMPLIED CORRELATION")
    print("─" * 70)
    print("\n  Modello: equicorrelazione  R_ij = ρ per i≠j,  R_ii = 1")

    # Usa ρ medio dalla matrice storica come ρ_vero del test sintetico
    N = len(S0)
    offdiag  = corr_hist[~np.eye(N, dtype=bool)]
    rho_true = float(offdiag.mean())
    print(f"  ρ_medio storico (usato come ρ_vero nel test): {rho_true:.4f}")

    K_list, C_market, rho_single, res_lsq = analyze_inverse_problem(
        S0, sigma, rho_true, r, T,
        MONEYNESS_GRID, N_PATHS_INVERSE, SEED
    )

    # ─────────────────────────────────────────────────────────────────────────
    # GRAFICI
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[C]  Generazione grafici ...")
    plot_extensions(
        geo_price, ref_arith, records_arith,
        K_list, C_market, rho_true, rho_single, res_lsq,
        S0, sigma, r, T, MONEYNESS_GRID,
        TICKERS, output_dir=script_dir,
    )

    print("\n" + "=" * 70)
    print("  RIEPILOGO FINALE")
    print("=" * 70)
    print(f"  Prezzo geometrico (analitico, lower bound): {geo_price:.6f}")
    print(f"  Prezzo aritmetico (stima MC riferimento):   {ref_arith:.6f}  "
          f"(+{(ref_arith - geo_price)/geo_price:.3%} rispetto al geometrico)")
    print(f"  ρ_vero del test inverso:   {rho_true:.4f}")
    print(f"  ρ_recuperata (singolo K):  {rho_single:.4f}")
    print(f"  ρ_recuperata (multi-K):    {res_lsq['rho']:.4f}")
    print("=" * 70 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # PARTE C: Analisi di sensitività al rumore
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[C]  SENSITIVITÀ AL RUMORE — quanto è stabile l'inversione?")
    print("─" * 70)

    noise_records = noise_sensitivity_analysis(
        S0, sigma, rho_true, r, T,
        MONEYNESS_GRID, N_PATHS_INVERSE, SEED
    )

    print("\n[D]  Generazione grafici sensitività ...")
    plot_noise_sensitivity(noise_records, rho_true, output_dir=script_dir)


if __name__ == "__main__":
    main()
