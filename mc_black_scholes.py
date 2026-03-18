#!/usr/bin/env python3
"""
================================================================================
  Black-Scholes Multidimensionale — Monte Carlo vs Soluzione Analitica
  Opzione Call Europea su Paniere Geometrico  (N asset generico)
================================================================================

FONDAMENTI TEORICI
------------------
Sotto la misura risk-neutral, N asset seguono GBM correlati:

    dSᵢ = r Sᵢ dt + σᵢ Sᵢ dWᵢ,    dWᵢ dWⱼ = ρᵢⱼ dt

Il paniere geometrico è:

    G_T = ( ∏ᵢ Sᵢ(T) )^(1/N)

Essendo il prodotto di log-normali ancora log-normale, log G_T ~ N(m, v²):

    m  = (1/N) Σᵢ [ log Sᵢ(0) + (r − σᵢ²/2) T ]
    v² = (T / N²) Σᵢ Σⱼ ρᵢⱼ σᵢ σⱼ  =  (T / N²) σᵀ R σ

La call europea su G ha soluzione analitica esatta (Gentle, 1993):

    C = e^{−rT} [ e^{m + v²/2} Φ(d₁)  −  K Φ(d₂) ]
    d₁ = (m − ln K + v²) / v
    d₂ = d₁ − v

Il Monte Carlo simula i percorsi tramite decomposizione di Cholesky di R per
riprodurre le correlazioni:  W = L Z,  Z ~ N(0,I),  R = L Lᵀ

La convergenza attesa dell'errore MC è  O(1/√N),  verificabile in scala log-log.

DATI
----
σᵢ e ρᵢⱼ vengono calibrati su rendimenti storici scaricati da Yahoo Finance
via yfinance e memorizzati in cache su disco (CACHE_FILE).
Se il download fallisce vengono usati parametri sintetici di fallback.

Riferimenti:
  - Gentle D. (1993). "Basket Weaving". Risk Magazine, 6(6), 51–52.
  - Carmona R. & Durrleman V. (2003). "Pricing and hedging spread options".
    SIAM Review, 45(4), 627–685.
================================================================================
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE  (modifica qui per cambiare asset / parametri)
# ─────────────────────────────────────────────────────────────────────────────

TICKERS       = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]   # N asset
T             = 1.0       # Maturità [anni]
r             = 0.05      # Tasso risk-free annualizzato
K_MONEYNESS   = 1.0       # Strike = K_MONEYNESS × G₀  (1.0 = ATM)
LOOKBACK_DAYS = 504       # Giorni di trading storici (~2 anni)

# Dimensioni di simulazione MC per l'analisi di convergenza
MC_SIZES  = [500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
N_REPEATS = 30            # Ripetizioni indipendenti per stimare la varianza MC

CACHE_FILE = "market_data_cache.pkl"
SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DOWNLOAD E CALIBRAZIONE
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_yfinance():
    """Installa yfinance se non presente."""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        print("  [pip] Installo yfinance ...")
        os.system(f"{sys.executable} -m pip install yfinance --quiet")
        import yfinance as yf
        return yf


def download_or_load(tickers: list, lookback_days: int, cache_file: str) -> pd.DataFrame | None:
    """
    Scarica i prezzi di chiusura aggiustati o li carica dalla cache locale.
    Restituisce None se il download fallisce (si useranno parametri sintetici).
    """
    if os.path.exists(cache_file):
        print(f"  [cache]  Trovato '{cache_file}', carico dati dal disco.")
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    yf = _ensure_yfinance()
    end   = datetime.today()
    start = end - timedelta(days=int(lookback_days * 365 / 252) + 90)

    print(f"  [yfinance] Download {tickers}  ({start.date()} → {end.date()}) ...")
    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )["Close"]
        raw = raw[tickers].dropna()
        raw = raw.iloc[-lookback_days:]
        assert len(raw) >= 60, "Dati storici insufficienti (< 60 righe)."
        with open(cache_file, "wb") as fh:
            pickle.dump(raw, fh)
        print(f"  [yfinance] OK — {len(raw)} giorni × {len(tickers)} asset → cache salvata.")
        return raw
    except Exception as exc:
        print(f"  [WARN] Download fallito: {exc}")
        print("         Utilizzo parametri sintetici di fallback.")
        return None


def calibrate(prices: pd.DataFrame):
    """
    Stima volatilità annualizzate e matrice di correlazione dai log-rendimenti.

    Returns
    -------
    S0   : ndarray (N,)  – ultimo prezzo osservato
    sigma: ndarray (N,)  – volatilità annualizzata
    corr : ndarray (N,N) – matrice di correlazione storica
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    sigma   = log_ret.std().values * np.sqrt(252)
    corr    = log_ret.corr().values
    S0      = prices.iloc[-1].values.astype(float)
    return S0, sigma, corr


def synthetic_params(n: int):
    """
    Parametri sintetici per N asset quando Yahoo Finance non è disponibile.
    Genera una matrice di correlazione valida (definita positiva) tramite
    il metodo del fattore comune.
    """
    rng   = np.random.default_rng(SEED)
    S0    = np.ones(n) * 100.0
    sigma = rng.uniform(0.15, 0.40, n)

    # Correlazione via modello fattoriale: ρᵢⱼ = βᵢ βⱼ / (‖β‖²) + ε δᵢⱼ
    beta  = rng.uniform(0.3, 0.8, n)
    corr  = np.outer(beta, beta)
    corr /= corr.max()                       # normalizza i fuori-diagonale < 1
    np.fill_diagonal(corr, 1.0)
    return S0, sigma, corr


def ensure_psd(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Aggiunge una perturbazione diagonale minima per garantire che la matrice
    di correlazione sia definita positiva (necessario per Cholesky).
    """
    eigmin = np.linalg.eigvalsh(corr).min()
    if eigmin < eps:
        shift = eps - eigmin + 1e-8
        corr  = corr + shift * np.eye(len(corr))
        # Ri-normalizza la diagonale a 1
        d    = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        print(f"  [FIX] Matrice di correlazione corretta PSD (shift={shift:.2e})")
    return corr


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SOLUZIONE ANALITICA — Geometric Basket European Call
# ─────────────────────────────────────────────────────────────────────────────

def geometric_basket_analytical(
    S0: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    K: float,
    r: float,
    T: float,
) -> float:
    """
    Prezzo analitico esatto della call europea su paniere geometrico.

    Sotto la misura risk-neutral:
        log G_T ~ N(m, v²)
    con:
        m  = (1/N) Σᵢ [ log Sᵢ(0) + (r − σᵢ²/2) T ]
        v² = (T / N²) σᵀ R σ

    Prezzo (formula Black-Scholes per variabile log-normale):
        C = e^{−rT} [ e^{m + v²/2} Φ(d₁) − K Φ(d₂) ]
        d₁ = (m − ln K + v²) / v
        d₂ = d₁ − v
    """
    N  = len(S0)
    m  = np.mean(np.log(S0) + (r - 0.5 * sigma**2) * T)
    v2 = (T / N**2) * float(sigma @ corr @ sigma)   # σᵀ R σ · T / N²
    v  = np.sqrt(max(v2, 1e-14))

    d1 = (m - np.log(K) + v2) / v
    d2 = d1 - v

    price = np.exp(-r * T) * (
        np.exp(m + 0.5 * v2) * norm.cdf(d1) - K * norm.cdf(d2)
    )
    return float(price)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

def geometric_basket_mc(
    S0: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    K: float,
    r: float,
    T: float,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Stima Monte Carlo del prezzo della call su paniere geometrico.

    Simulazione esatta (non per passi temporali) dei log-prezzi terminali:

        log Sᵢ(T) = log Sᵢ(0) + (r − σᵢ²/2)T + σᵢ √T Zᵢ

    con Z ~ N(0, R) ottenuto tramite decomposizione di Cholesky:
        R = L Lᵀ  →  W = L Z_iid,   Z_iid ~ N(0, I)

    Returns
    -------
    price : float  – prezzo MC scontato (media campionaria)
    se    : float  – errore standard della media
    """
    N  = len(S0)
    L  = cholesky(corr, lower=True)                    # R = L Lᵀ

    # Genera variabili standard i.i.d. → correla tramite L
    Z_iid = rng.standard_normal((N, n_paths))          # (N, n_paths)
    W     = L @ Z_iid                                  # (N, n_paths)  correlate

    # Log-prezzi terminali: (N, n_paths)
    log_ST = (
        np.log(S0) + (r - 0.5 * sigma**2) * T
    )[:, None] + sigma[:, None] * np.sqrt(T) * W

    # Paniere geometrico: media geometrica in log-spazio
    log_G  = log_ST.mean(axis=0)                       # (n_paths,)
    payoff = np.maximum(np.exp(log_G) - K, 0.0)

    disc  = np.exp(-r * T)
    price = disc * payoff.mean()
    se    = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return float(price), float(se)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ANALISI DI CONVERGENZA
# ─────────────────────────────────────────────────────────────────────────────

def convergence_analysis(
    S0, sigma, corr, K, r, T,
    mc_sizes: list,
    n_repeats: int,
    seed: int,
) -> tuple[float, list]:
    """
    Per ogni dimensione in mc_sizes esegue n_repeats run indipendenti e
    raccoglie medie, std e statistiche di errore rispetto al prezzo analitico.

    Returns
    -------
    true_price : float
    records    : list of dict  (una entry per ogni n in mc_sizes)
    """
    true_price = geometric_basket_analytical(S0, sigma, corr, K, r, T)

    print(f"\n  Prezzo analitico (riferimento):  {true_price:.8f}")
    print(f"\n  {'N paths':>10}  {'Media MC':>12}  {'Std MC':>12}  "
          f"{'|Errore|':>12}  {'Err relativo':>13}")
    print("  " + "─" * 65)

    records = []
    for n in mc_sizes:
        prices = []
        for rep in range(n_repeats):
            # Seed deterministico ma diverso per ogni (n, rep)
            rng = np.random.default_rng(seed + rep * 1009 + n % 100_000)
            p, _ = geometric_basket_mc(S0, sigma, corr, K, r, T, n, rng)
            prices.append(p)

        prices   = np.array(prices)
        mean_p   = prices.mean()
        std_p    = prices.std(ddof=1)
        abs_err  = abs(mean_p - true_price)
        rel_err  = abs_err / true_price

        print(f"  {n:>10,}  {mean_p:>12.8f}  {std_p:>12.8f}  "
              f"{abs_err:>12.8f}  {rel_err:>12.5%}")

        records.append({
            "n_paths":    n,
            "mean_price": mean_p,
            "std_price":  std_p,
            "abs_error":  abs_err,
            "rel_error":  rel_err,
            "prices":     prices,
        })

    return true_price, records


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALIZZAZIONE
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(true_price: float, records: list, tickers: list, output_dir: Path):
    """
    Genera tre pannelli:
      (A) Prezzo MC ± 2σ vs N simulazioni, con linea analitica
      (B) |Errore assoluto| in scala log-log, con riferimento 1/√N
      (C) Distribuzione delle stime MC per il run più grande
    """
    ns       = [rec["n_paths"]    for rec in records]
    means    = [rec["mean_price"] for rec in records]
    stds     = [rec["std_price"]  for rec in records]
    abs_errs = [rec["abs_error"]  for rec in records]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Monte Carlo vs Analitico — Geometric Basket European Call\n"
        f"Asset: {', '.join(tickers)}   |   T = {T} y   r = {r:.0%}   K = ATM",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Pannello A: convergenza del prezzo ──────────────────────────────────
    ax = axes[0]
    upper = [m + 2 * s for m, s in zip(means, stds)]
    lower = [m - 2 * s for m, s in zip(means, stds)]
    ax.fill_between(ns, lower, upper, alpha=0.22, color="steelblue", label="±2σ MC")
    ax.plot(ns, means, "o-", color="steelblue", lw=2, ms=5, label="Media MC")
    ax.axhline(true_price, color="crimson", lw=2, ls="--", label="Analitico (esatto)")
    ax.set_xscale("log")
    ax.set_xlabel("N simulazioni (log)", fontsize=11)
    ax.set_ylabel("Prezzo opzione", fontsize=11)
    ax.set_title("(A)  Convergenza del prezzo", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    # ── Pannello B: errore in log-log ────────────────────────────────────────
    ax = axes[1]
    ax.loglog(ns, abs_errs, "s-", color="darkorange", lw=2, ms=6,
              label="|Errore assoluto|")
    # Riferimento teorico  1/√N  calibrato sul primo punto
    c_ref = abs_errs[0] * np.sqrt(ns[0])
    ref_y = [c_ref / np.sqrt(n) for n in ns]
    ax.loglog(ns, ref_y, "k--", lw=1.5, label=r"$\propto 1/\sqrt{N}$ (teorico)")
    ax.set_xlabel("N simulazioni (log)", fontsize=11)
    ax.set_ylabel("|Errore assoluto| (log)", fontsize=11)
    ax.set_title("(B)  Convergenza dell'errore  (log-log)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.35)

    # Annotazione pendenza empirica
    log_ns  = np.log10(ns)
    log_err = np.log10(abs_errs)
    slope   = np.polyfit(log_ns, log_err, 1)[0]
    ax.annotate(
        f"pendenza empirica: {slope:.2f}\n(teorica: −0.50)",
        xy=(ns[len(ns)//2], abs_errs[len(ns)//2]),
        xytext=(0.35, 0.82), textcoords="axes fraction",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )

    # ── Pannello C: distribuzione stime MC (run più grande) ─────────────────
    ax = axes[2]
    last = records[-1]
    ax.hist(last["prices"], bins=30, color="mediumseagreen",
            edgecolor="white", density=True, alpha=0.85)
    ax.axvline(true_price, color="crimson", lw=2.5, ls="--",
               label=f"Analitico: {true_price:.5f}")
    ax.axvline(last["mean_price"], color="navy", lw=1.5, ls=":",
               label=f"Media MC:  {last['mean_price']:.5f}")
    ax.set_xlabel("Prezzo stimato", fontsize=11)
    ax.set_ylabel("Densità", fontsize=11)
    ax.set_title(
        f"(C)  Distribuzione delle {N_REPEATS} stime MC\n"
        f"(N = {last['n_paths']:,} paths per run)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    out_path = output_dir / "convergence_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot salvato in '{out_path}'")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Black-Scholes Multidim — Monte Carlo vs Analitico")
    print("  Geometric Basket European Call Option")
    print("=" * 70)

    # ── 1. Parametri di mercato ───────────────────────────────────────────────
    print(f"\n[1] Recupero dati di mercato  ({TICKERS})")
    script_dir = Path(__file__).parent if "__file__" in dir() else Path(".")
    cache_path = script_dir / CACHE_FILE

    prices = download_or_load(TICKERS, LOOKBACK_DAYS, str(cache_path))

    if prices is not None:
        S0, sigma, corr = calibrate(prices)
        source = "storici (Yahoo Finance)"
    else:
        N = len(TICKERS)
        S0, sigma, corr = synthetic_params(N)
        source = "sintetici (fallback)"

    corr = ensure_psd(corr)

    print(f"\n  Fonte parametri  : {source}")
    print(f"  Asset            : {TICKERS}")
    print(f"  Prezzi S₀        : {np.round(S0, 2)}")
    print(f"  Vol. annualizzate: {np.round(sigma, 4)}")
    print(f"\n  Matrice di correlazione:")
    df_corr = pd.DataFrame(corr, index=TICKERS, columns=TICKERS)
    print(df_corr.round(3).to_string())

    # ── 2. Parametri dell'opzione ─────────────────────────────────────────────
    G0 = float(np.exp(np.mean(np.log(S0))))
    K  = K_MONEYNESS * G0

    print(f"\n[2] Parametri opzione")
    print(f"  Tipo             : European Call su Paniere Geometrico")
    print(f"  G₀ (paniere geo) : {G0:.4f}")
    print(f"  Strike K         : {K:.4f}  (moneyness = {K_MONEYNESS:.2f}  →  ATM)")
    print(f"  Maturità T       : {T} anni")
    print(f"  Tasso risk-free  : {r:.2%}")
    print(f"  N asset          : {len(TICKERS)}")

    # ── 3. Analisi di convergenza ─────────────────────────────────────────────
    print("\n[3] Analisi di convergenza MC")
    print(f"    MC sizes   : {MC_SIZES}")
    print(f"    Ripetizioni: {N_REPEATS} per dimensione")

    true_price, records = convergence_analysis(
        S0, sigma, corr, K, r, T, MC_SIZES, N_REPEATS, SEED
    )

    # ── 4. Tabella di riepilogo ───────────────────────────────────────────────
    print("\n[4] Riepilogo statistiche di errore")
    print(f"\n  {'N paths':>10}  {'Media MC':>12}  {'Err. rel.':>11}  "
          f"{'Std MC':>12}  {'IC 95% (±)':>12}")
    print("  " + "─" * 65)
    for rec in records:
        ci = 1.96 * rec["std_price"]
        print(f"  {rec['n_paths']:>10,}  {rec['mean_price']:>12.6f}  "
              f"{rec['rel_error']:>10.4%}  {rec['std_price']:>12.6f}  "
              f"{ci:>12.6f}")

    # ── 5. Grafici ────────────────────────────────────────────────────────────
    print("\n[5] Generazione grafici ...")
    plot_results(true_price, records, TICKERS, output_dir=script_dir)

    print("\n" + "=" * 70)
    print(f"  Prezzo analitico esatto : {true_price:.8f}")
    best = records[-1]
    print(f"  Miglior stima MC        : {best['mean_price']:.8f}  "
          f"(N={best['n_paths']:,},  err rel = {best['rel_error']:.5%})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
