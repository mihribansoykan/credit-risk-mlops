"""Data-drift monitoring with the Population Stability Index (PSI).

PSI is the industry-standard drift metric in credit risk:
    PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)
Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift
(investigate the source and consider retraining).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_EPS = 1e-4


def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return np.clip(counts / max(len(values), 1), _EPS, None)


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = _bin_edges(expected, n_bins)
    e, a = _proportions(expected, edges), _proportions(actual, edges)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


def build_reference_profile(features: pd.DataFrame, n_bins: int = 10) -> dict:
    """Store bin edges + expected proportions from training data for later comparison."""
    profile = {}
    for col in features.columns:
        values = features[col].to_numpy(dtype=float)
        edges = _bin_edges(values, n_bins)
        profile[col] = {
            "edges": [float(x) for x in edges[1:-1]],  # +/-inf are implicit
            "expected": _proportions(values, edges).tolist(),
        }
    return profile


def save_profile(profile: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_profile(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def drift_report(profile: dict, features: pd.DataFrame) -> pd.DataFrame:
    """PSI of each feature in ``features`` against the stored reference profile."""
    rows = []
    for col, ref in profile.items():
        if col not in features.columns:
            continue
        edges = np.array([-np.inf, *ref["edges"], np.inf])
        e = np.asarray(ref["expected"])
        a = _proportions(features[col].to_numpy(dtype=float), edges)
        value = float(np.sum((a - e) * np.log(a / e)))
        rows.append({"feature": col, "psi": value, "status": psi_status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
