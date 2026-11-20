#!/usr/bin/env python3
"""
plot_theory_kappa_c.py

Grafica únicamente la curva TEÓRICA kappa_c(N) vs 1/N de Mertens (2000),
Eq. (32) -- sin superponer datos experimentales -- manteniendo el mismo
estilo visual que plot_k_vs_N.py.

Uso:
    python plot_theory_kappa_c.py                       # guarda PNG, sin display
    python plot_theory_kappa_c.py --show                # visualización interactiva
    python plot_theory_kappa_c.py --N-min 2 --N-max 200 # rango de N
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FOLDER_PATH = Path(__file__).parent / "Experiments"


# ==============================================================================
# 1. CURVA TEÓRICA
# ==============================================================================

def kappa_c_theory(N):
    """
    kappa_c(N) = 1 - ln(pi*N/6) / (2*N*ln 2), Eq. (32) de Mertens (2000),
    "The Easiest Hard Problem: Number Partitioning".
    """
    N = np.asarray(N, dtype=float)
    return 1.0 - np.log(np.pi * N / 6.0) / (2.0 * N * np.log(2.0))


# ==============================================================================
# 2. ESTILO (idéntico a plot_k_vs_N.py)
# ==============================================================================

def style_axes(ax):
    ax.set_xlabel(r"$1/N$", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$\kappa$", fontsize=14, fontweight="bold")
    ax.set_title(
        "Phase Transition in Number Partitioning\n"
        r"Theoretical Critical Bit-Length $\kappa_c$ vs. Problem Size $N$",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.1, linestyle=":", linewidth=0.5)


def _safe_xlim(ax, values, margin_frac=0.05, min_margin=1e-3):
    lo, hi = float(np.min(values)), float(np.max(values))
    margin = (hi - lo) * margin_frac
    if margin <= 0:
        margin = max(abs(lo) * margin_frac, min_margin)
    ax.set_xlim(lo - margin, hi + margin)


# ==============================================================================
# 3. PLOT
# ==============================================================================

def plot_theory_only(N_min=2, N_max=200, show=True, save=True, output_path=None):
    """
    Dibuja la curva teórica kappa_c(N) sobre 1/N usando el mismo estilo que
    plot_k_vs_N.py (línea azul, lw=2.5, leyenda con la fórmula, grid, etc.).
    """
    N = np.linspace(N_min, N_max, 1000)
    kappa = kappa_c_theory(N)
    inv_N = 1.0 / N

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(inv_N, kappa, color="blue", linewidth=2.5,
            label=r"$\kappa_c(N) = 1 - \frac{\ln(\pi N/6)}{2N\ln 2}$",
            zorder=3)

    style_axes(ax)
    _safe_xlim(ax, inv_N)

    y_margin = max((kappa.max() - kappa.min()) * 0.05, 0.005)
    ax.set_ylim(kappa.min() - y_margin, kappa.max() + y_margin)
    ax.legend(fontsize=10, loc="best", framealpha=0.95)

    plt.tight_layout()

    if save:
        if output_path is None:
            FOLDER_PATH.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = FOLDER_PATH / f"npp_theory_kappa_c_{ts}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved: {output_path}")

    if show:
        plt.show()

    return fig, ax


# ==============================================================================
# 4. CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot the theoretical kappa_c(N) curve from Mertens (2000) Eq. (32).",
    )
    parser.add_argument("--N-min", type=int, default=2,
                        help="Minimum N (default: 2)")
    parser.add_argument("--N-max", type=int, default=200,
                        help="Maximum N (default: 200)")
    parser.add_argument("--show", "-s", action="store_true",
                        help="Display the plot interactively (requires a GUI backend)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the plot to PNG")
    args = parser.parse_args()

    matplotlib.use("TkAgg" if args.show else "Agg")

    plot_theory_only(N_min=args.N_min, N_max=args.N_max,
                     show=args.show, save=not args.no_save)

    if not args.show:
        print("(Use --show flag to display interactively)")
    print("✓ Done.")


if __name__ == "__main__":
    main()