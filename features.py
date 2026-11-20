"""Shared, deterministic featurisation helpers used by all four ICONN-2027 studies.

Only real compositions/structures from public databases are featurised here;
no records are generated, interpolated or augmented.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data_raw")
CACHE = os.path.join(ROOT, "data_cache")
os.makedirs(CACHE, exist_ok=True)


def _comp_featurizer():
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (ElementProperty, Stoichiometry,
                                                  ValenceOrbital)
    return MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),   # Ward et al. 2016 (arXiv:1606.09551)
        Stoichiometry(),
        ValenceOrbital(props=["avg", "frac"]),
    ])


def physics_features(comps: pd.Series) -> pd.DataFrame:
    """Hand-built, physically motivated descriptors (electronegativity spread,
    ionic character, d/f-electron presence, count of anions)."""
    from pymatgen.core import Element
    anions = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P", "As", "Sb", "H"}
    rows = []
    for c in comps:
        els = [Element(e.symbol) for e in c.elements]
        fr = np.array([c.get_atomic_fraction(e) for e in els])
        X = np.array([e.X if e.X == e.X else np.nan for e in els], dtype=float)
        xmax, xmin = np.nanmax(X), np.nanmin(X)
        dX = xmax - xmin
        rows.append({
            "phys_dX": dX,
            "phys_ionic_char": 1 - np.exp(-0.25 * dX ** 2),  # Pauling ionic character
            "phys_wX_std": float(np.sqrt(np.nansum(fr * (X - np.nansum(fr * X)) ** 2))),
            "phys_frac_anion": float(sum(f for e, f in zip(els, fr) if e.symbol in anions)),
            "phys_frac_tm": float(sum(f for e, f in zip(els, fr) if e.is_transition_metal)),
            "phys_frac_f_block": float(sum(f for e, f in zip(els, fr) if e.is_lanthanoid or e.is_actinoid)),
            "phys_frac_metal": float(sum(f for e, f in zip(els, fr) if e.is_metal)),
            "phys_n_elements": len(els),
        })
    return pd.DataFrame(rows, index=comps.index)


def composition_features(formulas: pd.Series, cache_name: str | None = None,
                         n_jobs: int = 6) -> pd.DataFrame:
    """Magpie + stoichiometry + valence-orbital + physics descriptors."""
    from pymatgen.core import Composition
    path = os.path.join(CACHE, f"{cache_name}.pkl") if cache_name else None
    if path and os.path.exists(path):
        return pd.read_pickle(path)
    comps = formulas.apply(lambda f: Composition(f))
    df = pd.DataFrame({"composition": comps})
    fz = _comp_featurizer()
    fz.set_n_jobs(n_jobs)
    feats = fz.featurize_dataframe(df, "composition", ignore_errors=True, pbar=False)
    feats = feats.drop(columns=["composition"])
    feats = pd.concat([feats, physics_features(comps)], axis=1)
    feats.columns = [str(c).replace(" ", "_") for c in feats.columns]
    feats = feats.apply(pd.to_numeric, errors="coerce")
    if path:
        feats.to_pickle(path)
    return feats


def chem_system(formula: str) -> str:
    from pymatgen.core import Composition
    return "-".join(sorted(e.symbol for e in Composition(formula).elements))


def reduced_formula(formula: str) -> str:
    """Reduced formula, or None if the string is not a valid composition of real elements."""
    from pymatgen.core import Composition, Element
    try:
        c = Composition(formula)
        for e in c.elements:
            Element(e.symbol)
        return c.reduced_formula
    except Exception:
        return None


def regression_metrics(y, p) -> dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "Pearson_r": float(pearsonr(y, p)[0]),
        "Spearman_rho": float(spearmanr(y, p)[0]),
        "n": int(len(y)),
    }


def setup_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})
    return plt


def parity_plot(plt, y, p, path, title, unit="eV", label=None):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(y, p, s=6, alpha=0.45, color="#2b6cb0", edgecolors="none")
    lo, hi = float(min(np.min(y), np.min(p))), float(max(np.max(y), np.max(p)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    m = regression_metrics(y, p)
    ax.text(0.04, 0.96, f"MAE = {m['MAE']:.3f} {unit}\nRMSE = {m['RMSE']:.3f} {unit}\nR² = {m['R2']:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    ax.set_xlabel(f"Reference {label or 'value'} ({unit})")
    ax.set_ylabel(f"Predicted {label or 'value'} ({unit})")
    ax.set_title(title, fontsize=10)
    fig.savefig(path)
    plt.close(fig)
