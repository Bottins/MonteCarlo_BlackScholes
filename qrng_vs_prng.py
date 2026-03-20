#!/usr/bin/env python3
"""
================================================================================
  PRNG vs QRNG — Studio comparativo nel Monte Carlo Black-Scholes
================================================================================

Confronta il generatore pseudo-casuale di NumPy (PCG64) con numeri
quantisticamente casuali (QRNG) estratti da un file hardware di bit.

DOMANDE SCIENTIFICHE
--------------------
  1. La convergenza MC differisce tra PRNG e QRNG? (velocità, pendenza log-log)
  2. La varianza delle stime è minore con il QRNG?
  3. I due generatori superano le stesse suite di test statistici?
  4. La qualità del campionamento della distribuzione normale è equivalente?

FORMATO FILE QRNG
-----------------
  File di testo con soli caratteri '0' e '1' (senza spazi o newline).
  Conversione: ogni 32 bit consecutivi → uint32 big-endian → / 2^32 → U[0,1)
  Trasformazione a N(0,1): inverse-CDF (scipy.special.ndtri).

BUDGET QRNG (119,998,800 bit)
------------------------------
  Uniformi  : 3,749,962
  Normali   : 3,749,962
  Path max  : 749,992  (5 asset × N normali per path)

NOTA METODOLOGICA
-----------------
  PRNG: ogni ripetizione usa seed diverso → stime indipendenti.
  QRNG: ogni ripetizione usa il prossimo segmento non sovrapposto → no replacement.
  I primi 200,000 normali sono riservati ai plot di qualità; il campionatore
  MC parte da quella posizione.

Riferimento prezzo: European Call su paniere geometrico (Gentle 1993),
stessi parametri e cache di mc_black_scholes.py.
================================================================================
"""

import sys
import pickle
import warnings
from pathlib import Path

# Forza UTF-8 sul terminale Windows (CP1252 non supporta ─ ✓ ✗ ecc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.linalg import cholesky
from scipy.special import ndtri   # inverse-CDF normale, più veloce di norm.ppf

warnings.filterwarnings("ignore")


# ─── CONFIGURAZIONE ──────────────────────────────────────────────────────────

TICKERS     = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
T           = 1.0          # maturità [anni]
r           = 0.05         # tasso risk-free
K_MONEYNESS = 1.0          # ATM
SEED        = 42

QRNG_FILE  = "QNRG_pt1_120MByte.txt"
CACHE_FILE = "market_data_cache.pkl"

# Dimensioni MC per il confronto di convergenza
# Range ristretto al massimo raggiungibile dal QRNG (~10K paths con 200 rep);
# granularità fitta ai piccoli N per evidenziare la convergenza precoce del QRNG.
MC_SIZES  = [10, 50, 100, 500, 1_000, 5_000, 10_000]
N_REPEATS = 200             # ripetizioni PRNG (QRNG limitato dal pool)

DISPLAY_NORMALS = 200_000   # normali QRNG riservate ai grafici di qualità
BITS_PER_SAMPLE = 32        # bit QRNG per campione uniforme

COLORS = {"PRNG": "#1565C0", "QRNG": "#E64A19"}   # blu / arancio bruciato


# ─── 1.  LETTURA E CONVERSIONE FILE QRNG ─────────────────────────────────────

def load_qrng_bits(path: str) -> np.ndarray:
    """
    Legge il file QRNG e restituisce array uint8 di 0/1.
    Filtra automaticamente qualsiasi carattere non-'0'/non-'1'.
    """
    print(f"  Lettura: {path}")
    with open(path, "rb") as fh:
        raw = fh.read()
    arr  = np.frombuffer(raw, dtype=np.uint8)
    bits = arr[(arr == 48) | (arr == 49)] - 48   # '0'=48 → 0, '1'=49 → 1
    print(f"  Bit validi       : {len(bits):>15,}  ({len(bits) / 1e6:.1f} Mbit)")
    return bits


def bits_to_uniforms(bits: np.ndarray) -> np.ndarray:
    """
    Converte array di bit in valori uniformi [0, 1).

    Algoritmo efficiente in memoria:
      1. np.packbits (big-endian): 8 bit → 1 byte
      2. Reshape in gruppi di 4 byte: (N, 4)
      3. Ricostruzione uint32 big-endian con shift vettorializzato
      4. Normalizzazione / 2^32

    Usa packbits invece di operazioni per-bit per evitare l'allocazione
    di un array (N × 32) che sarebbe ~960 MB per il file completo.
    """
    n_vals = len(bits) // BITS_PER_SAMPLE
    packed = np.packbits(bits[: n_vals * BITS_PER_SAMPLE], bitorder="big")  # uint8
    p4     = packed.reshape(n_vals, 4)                   # view, nessuna copia
    shifts = np.array([24, 16, 8, 0], dtype=np.uint32)
    u32    = np.sum(p4.astype(np.uint32) << shifts, axis=1)
    return u32.astype(np.float64) / 2**32


def uniforms_to_normals(u: np.ndarray) -> np.ndarray:
    """
    Trasforma U[0,1) → N(0,1) via inverse-CDF (ppf).
    Clippa ai bordi per evitare ±∞.
    """
    eps = 0.5 / len(u)
    return ndtri(np.clip(u, eps, 1.0 - eps))


# ─── 2.  CAMPIONATORE QRNG ───────────────────────────────────────────────────

class QRNGSampler:
    """
    Pool di normali standard QRNG consumato sequenzialmente senza replacement.
    Ogni chiamata a .sample() avanza il puntatore interno.
    """

    def __init__(self, normals: np.ndarray, start: int = 0):
        self._pool = normals
        self._pos  = start
        avail = len(normals) - start
        print(f"  Normali QRNG disponibili per MC: {avail:>12,}"
              f"  (Path max 5 asset: {avail // len(TICKERS):,})")

    def sample(self, shape: tuple) -> np.ndarray:
        n = int(np.prod(shape))
        if self._pos + n > len(self._pool):
            raise RuntimeError(
                f"Pool QRNG esaurito — richiesti {n}, rimasti {self.remaining}"
            )
        out        = self._pool[self._pos : self._pos + n].reshape(shape)
        self._pos += n
        return out

    @property
    def remaining(self) -> int:
        return len(self._pool) - self._pos

    def max_paths(self, n_assets: int) -> int:
        return self.remaining // n_assets


# ─── 3.  PARAMETRI DI MERCATO ────────────────────────────────────────────────

def load_market_params():
    """Carica S0, σ, R dalla cache Yahoo Finance o usa valori sintetici."""
    cache = Path(CACHE_FILE)
    if cache.exists():
        with open(cache, "rb") as fh:
            prices = pickle.load(fh)
        log_ret = np.log(prices / prices.shift(1)).dropna()
        sigma   = log_ret.std().values * np.sqrt(252)
        corr    = log_ret.corr().values
        S0      = prices.iloc[-1].values.astype(float)
        source  = "Yahoo Finance (cache)"
    else:
        rng   = np.random.default_rng(SEED)
        N     = len(TICKERS)
        S0    = np.ones(N) * 100.0
        sigma = rng.uniform(0.15, 0.40, N)
        beta  = rng.uniform(0.3, 0.8, N)
        corr  = np.outer(beta, beta) / np.outer(beta, beta).max()
        np.fill_diagonal(corr, 1.0)
        source = "sintetici (fallback)"

    # Garantisce definita positiva per Cholesky
    eigmin = np.linalg.eigvalsh(corr).min()
    if eigmin < 1e-6:
        shift = 1e-6 - eigmin + 1e-8
        corr += shift * np.eye(len(corr))
        d     = np.sqrt(np.diag(corr))
        corr  = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)

    return S0, sigma, corr, source


# ─── 4.  PRICING ─────────────────────────────────────────────────────────────

def analytical_price(S0, sigma, corr, K, r, T) -> float:
    """Prezzo analitico — European Call su paniere geometrico (Gentle 1993)."""
    N  = len(S0)
    m  = np.mean(np.log(S0) + (r - 0.5 * sigma**2) * T)
    v2 = (T / N**2) * float(sigma @ corr @ sigma)
    v  = np.sqrt(max(v2, 1e-14))
    d1 = (m - np.log(K) + v2) / v
    d2 = d1 - v
    return float(
        np.exp(-r * T) * (
            np.exp(m + 0.5 * v2) * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)
        )
    )


def mc_price_from_Z(
    S0, sigma, corr, K, r, T, Z_iid: np.ndarray
) -> tuple[float, float]:
    """
    Prezzo MC da Z_iid di forma (N_assets, n_paths) — normali standard i.i.d.

    Simulazione esatta del log-prezzo terminale in un passo:
        log Sᵢ(T) = log Sᵢ(0) + (r − σᵢ²/2)T + σᵢ √T Wᵢ
    con W = L · Z_iid,  R = L Lᵀ  (Cholesky).
    """
    L      = cholesky(corr, lower=True)
    W      = L @ Z_iid
    log_ST = (
        (np.log(S0) + (r - 0.5 * sigma**2) * T)[:, None]
        + sigma[:, None] * np.sqrt(T) * W
    )
    log_G  = log_ST.mean(axis=0)
    payoff = np.maximum(np.exp(log_G) - K, 0.0)
    disc   = np.exp(-r * T)
    price  = disc * payoff.mean()
    se     = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return float(price), float(se)


# ─── 5.  ANALISI DI CONVERGENZA ──────────────────────────────────────────────

def run_convergence(
    label: str,
    S0, sigma, corr, K, r, T,
    mc_sizes: list,
    true_price: float,
    n_repeats: int = N_REPEATS,
    prng_seed: int = None,
    qrng: QRNGSampler = None,
) -> list[dict]:
    """
    Analisi di convergenza MC per sorgente PRNG o QRNG.

    PRNG: ogni ripetizione usa seed differente → stime indipendenti.
    QRNG: ogni ripetizione consuma il segmento successivo del pool.
          Il numero di ripetizioni è limitato dal pool rimanente.
    """
    N_A     = len(S0)
    records = []

    print(f"\n  [{label}]  {'N':>10}  {'rep':>5}  "
          f"{'Media':>12}  {'Std':>10}  {'|Err|':>12}  {'Err%':>10}")
    print(f"  [{label}]  " + "─" * 65)

    for n in mc_sizes:
        if qrng is not None:
            available_paths = qrng.max_paths(N_A)
            n_rep = min(available_paths // n if n > 0 else 0, n_repeats)
        else:
            n_rep = n_repeats

        if n_rep == 0:
            print(f"  [{label}]  {n:>10,}  — pool esaurito, stop —")
            break

        prices = []
        for rep in range(n_rep):
            if qrng is not None:
                Z = qrng.sample((N_A, n))
            else:
                rng = np.random.default_rng(prng_seed + rep * 1009 + n % 100_000)
                Z   = rng.standard_normal((N_A, n))
            p, _ = mc_price_from_Z(S0, sigma, corr, K, r, T, Z)
            prices.append(p)

        prices  = np.array(prices)
        mean_p  = prices.mean()
        std_p   = prices.std(ddof=1) if n_rep > 1 else np.nan
        abs_err = abs(mean_p - true_price)
        rel_err = abs_err / true_price

        std_s = f"{std_p:.6f}" if not np.isnan(std_p) else "       n/a"
        print(f"  [{label}]  {n:>10,}  {n_rep:>5}  "
              f"{mean_p:>12.6f}  {std_s:>10}  {abs_err:>12.6f}  {rel_err:>9.4%}")

        records.append(dict(
            n_paths=n, n_reps=n_rep,
            mean_price=mean_p, std_price=std_p,
            abs_error=abs_err, rel_error=rel_err,
            prices=prices, label=label,
        ))

    return records


# ─── 6.  SUITE DI TEST STATISTICI ────────────────────────────────────────────

def statistical_tests(
    label: str, uniforms: np.ndarray, bits: np.ndarray = None
) -> dict:
    """
    Suite di test di qualità su un campione di uniformi [0,1).

    Test implementati
    -----------------
    KS       Kolmogorov-Smirnov vs Uniform(0,1)    — uniformità distribuzione
    Chi²     Chi-square (20 bin equidistanti)        — frequenze assolute
    AC1      Autocorrelazione lag-1                  — indipendenza sequenziale
    Runs     Runs test sopra/sotto mediana           — pattern di alternanza
    AD       Anderson-Darling (dopo ppf)             — normalità della trasf.
    BitFreq  Frequenza bit "1" + Z-test (solo QRNG)  — balance hardware
    """
    MAX_N = 100_000
    u     = uniforms[:MAX_N]

    # 1. KS
    ks_s, ks_p = stats.kstest(u, "uniform")

    # 2. Chi-square
    obs, _       = np.histogram(u, bins=20, range=(0, 1))
    expected     = np.full(20, len(u) / 20.0)
    c2_s, c2_p   = stats.chisquare(obs, expected)

    # 3. Autocorrelazione lag-1
    ac1 = float(np.corrcoef(u[:-1], u[1:])[0, 1])
    thr = 3.0 / np.sqrt(len(u))

    # 4. Runs test (sopra/sotto mediana)
    above    = (u > np.median(u)).astype(int)
    n_runs   = 1 + int(np.sum(above[1:] != above[:-1]))
    n1, n2   = int(above.sum()), int(len(above) - above.sum())
    mu_r     = 2 * n1 * n2 / (n1 + n2) + 1
    var_r    = 2 * n1 * n2 * (2*n1*n2 - n1 - n2) / ((n1+n2)**2 * (n1+n2-1))
    runs_z   = (n_runs - mu_r) / np.sqrt(var_r)
    runs_p   = 2 * (1 - stats.norm.cdf(abs(runs_z)))

    # 5. Anderson-Darling sulla trasformazione normale (10K campioni)
    eps      = 0.5 / len(u)
    z_sample = ndtri(np.clip(u[:10_000], eps, 1 - eps))
    ad_s, ad_crit, _ = stats.anderson(z_sample, dist="norm")
    ad_pass  = bool(ad_s < ad_crit[2])   # livello 5%

    # 6. Frequenza bit (solo se si passano i bit grezzi)
    bf = None
    if bits is not None:
        n_b   = len(bits)
        freq1 = float(bits.sum()) / n_b
        z_bf  = (bits.sum() - n_b * 0.5) / np.sqrt(n_b * 0.25)
        p_bf  = 2 * (1 - stats.norm.cdf(abs(z_bf)))
        bf    = {"freq": freq1, "z": float(z_bf), "p": float(p_bf),
                 "pass": bool(p_bf > 0.01)}

    results = {
        "KS":      {"stat": ks_s,    "p": ks_p,    "pass": ks_p > 0.01},
        "Chi2":    {"stat": c2_s,    "p": c2_p,    "pass": c2_p > 0.01},
        "AC1":     {"stat": ac1,     "p": None,    "pass": abs(ac1) < thr},
        "Runs":    {"stat": runs_z,  "p": runs_p,  "pass": runs_p > 0.01},
        "AD-norm": {"stat": ad_s,    "p": None,    "pass": ad_pass},
        "BitFreq": bf,
    }

    # Stampa tabella
    print(f"\n  Test statistici — {label}  (campione {len(u):,})")
    print(f"  {'Test':<26} {'Statistica':>12} {'p-value':>12} {'Esito':>8}")
    print("  " + "─" * 62)
    rows = [
        ("KS vs Uniform(0,1)",   ks_s,   ks_p,    results["KS"]["pass"]),
        ("Chi-square (20 bin)",  c2_s,   c2_p,    results["Chi2"]["pass"]),
        ("Autocorr lag-1",       ac1,    None,    results["AC1"]["pass"]),
        ("Runs (sopra mediana)", runs_z, runs_p,  results["Runs"]["pass"]),
        ("Anderson-Darling",     ad_s,   None,    ad_pass),
    ]
    if bf:
        rows.append((
            f"Bit freq(1s)={bf['freq']:.5f}", bf["z"], bf["p"], bf["pass"]
        ))
    for name_t, st, pv, ok in rows:
        pv_s = f"{pv:>12.4f}" if pv is not None else "           —"
        ok_s = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name_t:<26} {st:>12.5f} {pv_s} {ok_s:>8}")

    return results


# ─── 7.  FIGURA 1: CONVERGENZA ───────────────────────────────────────────────

def plot_convergence(
    prng_recs: list, qrng_recs: list, true_price: float, output_path
):
    """
    2×2 — Confronto convergenza PRNG vs QRNG.

    (A) Prezzo MC ± 2σ vs N         (B) |Errore| log-log + ref 1/√N
    (C) Std stime MC vs N           (D) Distribuzione al N massimo disponibile
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "PRNG (NumPy PCG64) vs QRNG (hardware) — Monte Carlo Convergence\n"
        f"Geometric Basket European Call  |  T={T}y  r={r:.0%}  K=ATM  "
        f"N_asset={len(TICKERS)}  |  Analytical price = {true_price:.6f}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # ── helper ──────────────────────────────────────────────────────────────
    def _add_source(recs, lbl):
        if not recs:
            return
        c    = COLORS[lbl]
        ns   = [r["n_paths"]    for r in recs]
        mu   = [r["mean_price"] for r in recs]
        errs = [r["abs_error"]  for r in recs]
        stds = [r["std_price"]  for r in recs]

        # A: prezzo ± 2σ
        ax_a.plot(ns, mu, "o-", color=c, lw=2, ms=5, label=lbl)
        valid = [(n, m, s) for n, m, s in zip(ns, mu, stds) if not np.isnan(s)]
        if valid:
            nv, mv, sv = zip(*valid)
            ax_a.fill_between(nv,
                              [m - 2*s for m, s in zip(mv, sv)],
                              [m + 2*s for m, s in zip(mv, sv)],
                              alpha=0.18, color=c)

        # B: errore log-log
        ax_b.loglog(ns, errs, "s-", color=c, lw=2, ms=6, label=lbl)

        # C: std
        valid_std = [(n, s) for n, s in zip(ns, stds) if not np.isnan(s)]
        if len(valid_std) >= 2:
            nv2, sv2 = zip(*valid_std)
            ax_c.loglog(nv2, sv2, "D-", color=c, lw=2, ms=5, label=lbl)
            # pendenza empirica
            slope_std = np.polyfit(np.log10(nv2), np.log10(sv2), 1)[0]
            mid = len(nv2) // 2
            ax_c.annotate(
                f"{lbl}: {slope_std:.2f}",
                xy=(nv2[mid], sv2[mid]),
                xytext=(5, 5 if lbl == "PRNG" else -15),
                textcoords="offset points",
                fontsize=8, color=c,
            )

    _add_source(prng_recs, "PRNG")
    _add_source(qrng_recs, "QRNG")

    # Linea analitica — Panel A
    ax_a.axhline(true_price, color="crimson", lw=2, ls="--",
                 label=f"Analytical: {true_price:.5f}")

    # Riferimento 1/√N — Panel B
    all_recs = prng_recs + qrng_recs
    if all_recs:
        ns_ref = sorted(set(r["n_paths"] for r in all_recs))
        first  = min(all_recs, key=lambda x: x["n_paths"])
        c_ref  = first["abs_error"] * np.sqrt(first["n_paths"])
        ax_b.loglog(ns_ref, [c_ref / np.sqrt(n) for n in ns_ref],
                    "k--", lw=1.5, label=r"$1/\sqrt{N}$ (theoretical)")

    # Pendenza empirica dell'errore
    for recs, lbl in [(prng_recs, "PRNG"), (qrng_recs, "QRNG")]:
        if len(recs) >= 3:
            slope = np.polyfit(
                np.log10([r["n_paths"]   for r in recs]),
                np.log10([r["abs_error"] for r in recs]), 1
            )[0]
            mid = len(recs) // 2
            ax_b.annotate(
                f"{lbl}: {slope:.2f}",
                xy=(recs[mid]["n_paths"], recs[mid]["abs_error"]),
                xytext=(5, 5 if lbl == "PRNG" else -15),
                textcoords="offset points",
                fontsize=8, color=COLORS[lbl],
            )

    # Riferimento 1/√N — Panel C
    valid_prng_std = [(r["n_paths"], r["std_price"])
                      for r in prng_recs if not np.isnan(r["std_price"])]
    if valid_prng_std:
        nv0, sv0 = valid_prng_std[0]
        ns_c     = [v[0] for v in valid_prng_std]
        c_std    = sv0 * np.sqrt(nv0)
        ax_c.loglog(ns_c, [c_std / np.sqrt(n) for n in ns_c],
                    "k--", lw=1.5, label=r"$1/\sqrt{N}$")

    # Panel D: distribuzione stime al N massimo disponibile
    ax_d.axvline(true_price, color="crimson", lw=2.5, ls="--", zorder=5,
                 label=f"Analytical: {true_price:.5f}")
    for recs, lbl in [(prng_recs, "PRNG"), (qrng_recs, "QRNG")]:
        if recs:
            best = max(recs, key=lambda x: x["n_paths"])
            if len(best["prices"]) > 1:
                ax_d.hist(best["prices"], bins=25, color=COLORS[lbl],
                          alpha=0.60, density=True,
                          label=f"{lbl}  N={best['n_paths']:,}, {best['n_reps']} rep")

    # Formattazione
    for ax, title, xlabel, ylabel in [
        (ax_a, "(A)  Price convergence  (±2σ)",
         "N simulations", "Option price"),
        (ax_b, "(B)  Absolute error  (log-log scale)",
         "N simulations", "|Absolute error|"),
        (ax_c, r"(C)  Estimate std. deviation  (log-log scale)",
         "N simulations", "MC estimate std. dev."),
        (ax_d, "(D)  Estimate distribution at max N",
         "Estimated price", "Density"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.30)

    ax_a.set_xscale("log")
    ax_b.grid(True, which="both", alpha=0.30)
    ax_c.grid(True, which="both", alpha=0.30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure 1 salvata: '{output_path}'")
    plt.show()


# ─── 8.  FIGURA 2: QUALITÀ STATISTICA ────────────────────────────────────────

def plot_statistical_quality(
    prng_uniforms: np.ndarray,
    qrng_uniforms: np.ndarray,
    qrng_bits:     np.ndarray,
    prng_normals:  np.ndarray,
    qrng_normals:  np.ndarray,
    stat_prng:     dict,
    stat_qrng:     dict,
    output_path,
):
    """
    2×3 — Qualità statistica dei due generatori.

    (A) Istogrammi U[0,1)           (B) Autocorrelazione (lag 1-40)
    (C) Q-Q plot PRNG (normali)     (D) Q-Q plot QRNG (normali)
    (E) Frequenza bit QRNG rolling  (F) Riepilogo test statistici
    """
    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle(
        "Statistical Quality — PRNG (NumPy PCG64) vs QRNG (Hardware)",
        fontsize=13, fontweight="bold",
    )

    N_HIST  = 100_000
    N_ACF   = 10_000
    N_QQ    = 10_000
    N_LAGS  = 40

    # ── (A) Istogrammi U[0,1) ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for lbl, u in [("PRNG", prng_uniforms[:N_HIST]),
                   ("QRNG", qrng_uniforms[:N_HIST])]:
        ax_a.hist(u, bins=100, color=COLORS[lbl], alpha=0.60,
                  density=True, label=lbl)
    ax_a.axhline(1.0, color="k", ls="--", lw=1, label="Ideal U[0,1]")
    ax_a.set_xlabel("Value", fontsize=10)
    ax_a.set_ylabel("Density", fontsize=10)
    ax_a.set_title("(A)  Uniform Distribution", fontsize=11)
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.30)

    # ── (B) Autocorrelazione lag 1-40 ────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    for lbl, u in [("PRNG", prng_uniforms[:N_ACF]),
                   ("QRNG", qrng_uniforms[:N_ACF])]:
        acf = [float(np.corrcoef(u[:-lag], u[lag:])[0, 1])
               for lag in range(1, N_LAGS + 1)]
        ax_b.plot(range(1, N_LAGS + 1), acf, "o-",
                  color=COLORS[lbl], ms=3, lw=1.2, label=lbl)
    bound = 1.96 / np.sqrt(N_ACF)
    ax_b.axhline(+bound, color="gray", ls="--", lw=1,
                 label=f"±1.96/√N = ±{bound:.4f}")
    ax_b.axhline(-bound, color="gray", ls="--", lw=1)
    ax_b.axhline(0, color="k", lw=0.5)
    ax_b.set_xlabel("Lag", fontsize=10)
    ax_b.set_ylabel("Autocorrelation", fontsize=10)
    ax_b.set_title(f"(B)  Uniform ACF  (lag 1\u2013{N_LAGS})", fontsize=11)
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.30)

    # ── (C) Q-Q PRNG normale ─────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    z_p  = prng_normals[:N_QQ]
    (osm, osr), (slope, intercept, r_) = stats.probplot(z_p, dist="norm")
    ax_c.plot(osm, osr, ".", color=COLORS["PRNG"], ms=2, alpha=0.50)
    ax_c.plot(osm, slope * np.array(osm) + intercept, "k-", lw=1.5)
    ax_c.set_xlabel("Theoretical quantiles N(0,1)", fontsize=10)
    ax_c.set_ylabel("Sample quantiles", fontsize=10)
    ax_c.set_title(f"(C)  Q-Q PRNG  (R²={r_**2:.6f})", fontsize=11)
    ax_c.grid(True, alpha=0.30)

    # ── (D) Q-Q QRNG normale ─────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    z_q  = qrng_normals[:N_QQ]
    (osm, osr), (slope, intercept, r_) = stats.probplot(z_q, dist="norm")
    ax_d.plot(osm, osr, ".", color=COLORS["QRNG"], ms=2, alpha=0.50)
    ax_d.plot(osm, slope * np.array(osm) + intercept, "k-", lw=1.5)
    ax_d.set_xlabel("Theoretical quantiles N(0,1)", fontsize=10)
    ax_d.set_ylabel("Sample quantiles", fontsize=10)
    ax_d.set_title(f"(D)  Q-Q QRNG  (R²={r_**2:.6f})", fontsize=11)
    ax_d.grid(True, alpha=0.30)

    # ── (E) Frequenza bit QRNG — finestre scorrevoli ─────────────────────────
    ax_e  = fig.add_subplot(gs[1, 1])
    WIN   = 10_000
    n_win = len(qrng_bits) // WIN
    freqs = qrng_bits[: n_win * WIN].reshape(n_win, WIN).mean(axis=1)
    ax_e.plot(np.arange(n_win), freqs, color=COLORS["QRNG"], lw=0.6, alpha=0.85)
    ax_e.axhline(0.5, color="k", ls="--", lw=1.5, label="0.5 (ideal)")
    sigma_bf = 0.5 / np.sqrt(WIN)
    ax_e.axhline(0.5 + 3 * sigma_bf, color="red", ls=":", lw=1,
                 label=f"\u00b13\u03c3 = \u00b1{3*sigma_bf:.4f}")
    ax_e.axhline(0.5 - 3 * sigma_bf, color="red", ls=":", lw=1)
    ax_e.set_xlabel(f"Window (\u00d7{WIN:,} bits)", fontsize=10)
    ax_e.set_ylabel("Bit '1' frequency", fontsize=10)
    ax_e.set_title("(E)  QRNG bit frequency (windows {:.0f}K)".format(WIN/1e3),
                   fontsize=11)
    ax_e.legend(fontsize=9)
    ax_e.grid(True, alpha=0.30)

    # ── (F) Tabella riepilogativa test statistici ─────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    col_labels = ["Test", "PRNG", "QRNG"]
    test_keys  = [
        ("KS vs Uniform(0,1)", "KS"),
        ("Chi² (20 bin)",      "Chi2"),
        ("Autocorr lag-1",     "AC1"),
        ("Runs test",          "Runs"),
        ("Anderson-Darling",   "AD-norm"),
    ]
    rows = []
    for label_t, key in test_keys:
        p_ok = stat_prng[key]["pass"]
        q_ok = stat_qrng[key]["pass"]
        rows.append([
            label_t,
            "✓" if p_ok else "✗",
            "✓" if q_ok else "✗",
        ])
    bf = stat_qrng.get("BitFreq")
    if bf:
        rows.append([f"Bit freq = {bf['freq']:.4f}", "—",
                     "✓" if bf["pass"] else "✗"])

    tbl = ax_f.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.scale(1.3, 1.8)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#DDEEFF")
            cell.set_text_props(fontweight="bold")
        elif j in (1, 2):
            txt = cell.get_text().get_text()
            if txt == "✓":
                cell.set_facecolor("#C8E6C9")
            elif txt == "✗":
                cell.set_facecolor("#FFCDD2")
    ax_f.set_title("(F)  Statistical test summary  (\u03b1 = 1%)", fontsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Figure 2 salvata: '{output_path}'")
    plt.show()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  PRNG vs QRNG — Studio comparativo Monte Carlo Black-Scholes")
    print("=" * 72)

    script_dir = Path(__file__).parent

    # ── [1] Parametri di mercato ──────────────────────────────────────────────
    print("\n[1] Parametri di mercato")
    S0, sigma, corr, source = load_market_params()
    G0 = float(np.exp(np.mean(np.log(S0))))
    K  = K_MONEYNESS * G0

    print(f"  Fonte          : {source}")
    print(f"  Asset          : {TICKERS}")
    print(f"  S0             : {np.round(S0, 2)}")
    print(f"  Sigma ann.     : {np.round(sigma, 4)}")
    print(f"  G0             : {G0:.4f}   K (ATM) : {K:.4f}")
    print(f"\n  Matrice di correlazione:")
    print(pd.DataFrame(corr, index=TICKERS, columns=TICKERS).round(3).to_string())

    true_price = analytical_price(S0, sigma, corr, K, r, T)
    print(f"\n  Prezzo analitico esatto : {true_price:.8f}")

    # ── [2] Caricamento e conversione file QRNG ───────────────────────────────
    print("\n[2] Caricamento QRNG")
    bits     = load_qrng_bits(str(script_dir / QRNG_FILE))
    uniforms = bits_to_uniforms(bits)
    normals  = uniforms_to_normals(uniforms)
    print(f"  Uniformi generate: {len(uniforms):>12,}")
    print(f"  Normali generate : {len(normals):>12,}")

    # Campioni PRNG di riferimento (stessa dimensione, per confronto)
    prng_rng          = np.random.default_rng(SEED)
    prng_uniforms_ref = prng_rng.random(len(uniforms))
    prng_normals_ref  = np.random.default_rng(SEED + 1).standard_normal(len(normals))

    # ── [3] Test statistici di qualità ────────────────────────────────────────
    print("\n[3] Test statistici di qualità")
    stat_prng = statistical_tests("PRNG (PCG64)", prng_uniforms_ref, bits=None)
    stat_qrng = statistical_tests("QRNG (hardware)", uniforms, bits=bits)

    # ── [4] Analisi di convergenza ────────────────────────────────────────────
    print("\n[4] Analisi di convergenza MC")
    print(f"  MC sizes   : {MC_SIZES}")
    print(f"  Ripetizioni PRNG: {N_REPEATS} per dimensione")
    print(f"  Ripetizioni QRNG: auto (limitato dal pool, offset={DISPLAY_NORMALS:,})")

    # Il campionatore parte dopo i DISPLAY_NORMALS riservati ai grafici di qualità
    qrng_sampler = QRNGSampler(normals, start=DISPLAY_NORMALS)

    prng_records = run_convergence(
        "PRNG", S0, sigma, corr, K, r, T,
        MC_SIZES, true_price, N_REPEATS, prng_seed=SEED,
    )
    qrng_records = run_convergence(
        "QRNG", S0, sigma, corr, K, r, T,
        MC_SIZES, true_price, N_REPEATS, qrng=qrng_sampler,
    )

    # ── [5] Riepilogo comparativo ─────────────────────────────────────────────
    print("\n[5] Riepilogo comparativo  (prezzo analitico esatto = "
          f"{true_price:.8f})")
    print(f"\n  {'N':>10}  {'PRNG rep':>9}  {'PRNG Err%':>11}  "
          f"{'PRNG Std':>12}  {'QRNG rep':>9}  {'QRNG Err%':>11}  {'QRNG Std':>12}")
    print("  " + "─" * 80)
    qd = {r["n_paths"]: r for r in qrng_records}
    for rp in prng_records:
        n    = rp["n_paths"]
        rq   = qd.get(n)
        q_err = f"{rq['rel_error']:>11.4%}" if rq else "          —"
        q_std = f"{rq['std_price']:>12.6f}" if rq and not np.isnan(rq["std_price"]) else "           —"
        q_rep = f"{rq['n_reps']:>9}" if rq else "         —"
        p_std = f"{rp['std_price']:>12.6f}" if not np.isnan(rp["std_price"]) else "           —"
        print(f"  {n:>10,}  {rp['n_reps']:>9}  {rp['rel_error']:>11.4%}  "
              f"{p_std}  {q_rep}  {q_err}  {q_std}")

    # ── [6] Grafici ───────────────────────────────────────────────────────────
    print("\n[6] Generazione grafici ...")
    plot_convergence(
        prng_records, qrng_records, true_price,
        script_dir / "qrng_prng_convergence.png",
    )
    plot_statistical_quality(
        prng_uniforms_ref[:DISPLAY_NORMALS],
        uniforms[:DISPLAY_NORMALS],
        bits,
        prng_normals_ref[:DISPLAY_NORMALS],
        normals[:DISPLAY_NORMALS],
        stat_prng,
        stat_qrng,
        script_dir / "qrng_prng_quality.png",
    )

    print("\n" + "=" * 72)
    print("  Analisi completata.")
    print(f"  Output: qrng_prng_convergence.png")
    print(f"          qrng_prng_quality.png")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
