#!/usr/bin/env python3
"""
Equal-count (quantile count) binning utilities.

Provides three core functions:
- compute_equal_count_bins(values, n_bins)
- assign_bins(values, binspec)
- summarize_bins(values, y, binspec, weights=None)

Designed to be deterministic and robust to ties by using rank-based binning
on the reference data, and by deriving bin edges from the observed per‑bin
maxima. Edges are then reused to assign bins on new data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class BinSpec:
    """Specification of equal-count bins derived from a reference series.

    Attributes:
        n_bins: Number of bins.
        edges: Array of length n_bins+1 representing right-open intervals
               (edges[i], edges[i+1]] for bin i. edges[0] is -inf to admit
               any value less than or equal to the first bin max.
        lows:  Per-bin observed min (length n_bins; informational).
        highs: Per-bin observed max (length n_bins). highs is used to build
               edges; edges[1:] == highs (ascending by bin).
        centers: Per-bin mean value (length n_bins; informational).
        counts: Per-bin counts on the reference data (length n_bins).
    """

    n_bins: int
    edges: np.ndarray
    lows: np.ndarray
    highs: np.ndarray
    centers: np.ndarray
    counts: np.ndarray


def _to_series(x: Iterable) -> pd.Series:
    if isinstance(x, (pd.Series, pd.Index)):
        return pd.Series(x.values, index=None)
    return pd.Series(np.asarray(x))


def compute_equal_count_bins(values: Iterable, n_bins: int = 10) -> BinSpec:
    """Compute equal-count bins using rank-based binning on a reference series.

    - Drops NaNs in the reference values for bin construction.
    - Uses deterministic ranking (method='first') to split into near-equal counts.
    - Builds bin edges from per-bin maxima so that later assignment is fast via
      searchsorted on edges.

    Returns BinSpec with edges length n_bins+1 where bin i corresponds to
    (edges[i], edges[i+1]]. Values <= edges[1] fall into bin 0, and so on.
    """
    s = _to_series(values)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        raise ValueError("No valid (non-NaN) values to compute bins")
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    n = len(s)
    # Deterministic rank from 1..n then map to bins 0..n_bins-1
    ranks = s.rank(method="first")
    bin_idx = ((ranks - 1) * n_bins / n).astype(int).clip(0, n_bins - 1)
    # Aggregate per-bin stats (keep original order via groupby on a DataFrame)
    df = pd.DataFrame({"v": s.values, "bin": bin_idx.values})
    grp = df.groupby("bin", as_index=False)
    lows = grp["v"].min().set_index("bin")["v"]
    highs = grp["v"].max().set_index("bin")["v"]
    centers = grp["v"].mean().set_index("bin")["v"]
    # counts: use value_counts to avoid column name issues with GroupBy.size()
    counts = df["bin"].value_counts().sort_index()

    # Reindex in bin order and fill potential missing bins (if extreme ties)
    idx = pd.Index(range(n_bins), name="bin")
    lows = lows.reindex(idx, fill_value=lows.min() if len(lows) else np.nan)
    highs = highs.reindex(idx, method="ffill")
    centers = centers.reindex(idx, method="ffill")
    counts = counts.reindex(idx, fill_value=0)

    # Build edges as [-inf] + list(highs)
    edges = np.empty(n_bins + 1, dtype=float)
    edges[0] = -np.inf
    # Ensure monotonic non-decreasing highs
    highs_sorted = np.maximum.accumulate(highs.values.astype(float))
    edges[1:] = highs_sorted

    return BinSpec(
        n_bins=n_bins,
        edges=edges,
        lows=lows.values.astype(float),
        highs=highs_sorted,
        centers=centers.values.astype(float),
        counts=counts.values.astype(int),
    )


def assign_bins(values: Iterable, binspec: BinSpec, *, out_of_range: str = "clip") -> np.ndarray:
    """Assign values to bins using a precomputed BinSpec.

    Parameters:
        values: Array-like of numeric values to bucketize.
        binspec: BinSpec from compute_equal_count_bins.
        out_of_range: Behavior for values > last edge (or NaN):
            - 'clip' (default): clamp to last/first bin; NaNs -> -1.
            - 'error': raise ValueError on out-of-range or NaN.
            - 'ignore': return -1 for NaNs and values > last edge.

    Returns:
        np.ndarray[int] of bin indices in [0..n_bins-1]; NaNs/out-of-range per policy.
    """
    s = _to_series(values)
    x = pd.to_numeric(s, errors="coerce")
    # searchsorted with left side so that values equal to an edge
    # are assigned to the lower bin, matching (edges[i], edges[i+1]] semantics
    idx = np.searchsorted(binspec.edges, x.values, side="left") - 1
    # Out-of-range handling: idx can be n_bins if x > last edge; or -1 for NaNs
    mask_nan = x.isna().values
    mask_hi = idx >= binspec.n_bins
    mask_lo = idx < 0
    if out_of_range == "clip":
        idx[mask_hi] = binspec.n_bins - 1
        idx[mask_lo] = 0
        idx[mask_nan] = -1
    elif out_of_range == "ignore":
        idx[mask_hi | mask_lo | mask_nan] = -1
    else:  # 'error'
        if mask_nan.any() or mask_hi.any() or mask_lo.any():
            raise ValueError("Found NaN or out-of-range values during bin assignment")
    return idx.astype(int)


def summarize_bins(
    values: Iterable,
    y: Iterable,
    binspec: BinSpec,
    *,
    weights: Optional[Iterable] = None,
    include_value_stats: bool = True,
) -> pd.DataFrame:
    """Summarize y by equal-count bins defined on values.

    Parameters:
        values: Numeric array used for bin assignment (e.g., probs/scores).
        y: Numeric/binary array to aggregate within bins (e.g., target).
        binspec: BinSpec produced on a reference set.
        weights: Optional sample weights for weighted mean/count.
        include_value_stats: If True, include per-bin lows/highs/centers from binspec.

    Returns:
        DataFrame with columns:
          - bin: int [0..n_bins-1]
          - n: count of assigned samples in this call (unweighted)
          - mean_y: mean of y (weighted if weights provided)
          - w_sum: sum of weights (if provided)
          - p_low, p_high, p_center (if include_value_stats)
    """
    v = pd.to_numeric(_to_series(values), errors="coerce")
    yy = pd.to_numeric(_to_series(y), errors="coerce")
    if len(v) != len(yy):
        raise ValueError("values and y must have the same length")
    w = None if weights is None else pd.to_numeric(_to_series(weights), errors="coerce")
    idx = assign_bins(v, binspec, out_of_range="ignore")
    df = pd.DataFrame({"bin": idx, "y": yy, "w": w})
    df = df[df["bin"] >= 0]  # drop NaN/out-of-range

    if w is None:
        grp = df.groupby("bin")
        agg = grp.agg(n=("y", "size"), mean_y=("y", "mean")).reset_index()
        agg["w_sum"] = np.nan
    else:
        # weighted mean: sum(w*y)/sum(w)
        df = df.dropna(subset=["w"])  # drop rows with NaN weights
        grp = df.groupby("bin")
        sums = grp.agg(w_sum=("w", "sum"), wy_sum=("y", lambda s: float(np.dot(s, grp.get_group(s.name)["w"]))))
        # The above is clumsy due to groupby; compute robustly instead
        # Recompute with vectorized approach
        gb = df.groupby("bin")
        w_sum = gb["w"].sum()
        wy_sum = gb.apply(lambda g: float(np.dot(g["y"], g["w"])) )
        mean_y = wy_sum / w_sum.replace(0.0, np.nan)
        agg = pd.DataFrame({"bin": w_sum.index.values, "n": gb.size().values, "mean_y": mean_y.values, "w_sum": w_sum.values})

    # Attach value stats from binspec
    if include_value_stats:
        extra = pd.DataFrame(
            {
                "bin": np.arange(binspec.n_bins, dtype=int),
                "p_low": binspec.lows,
                "p_high": binspec.highs,
                "p_center": binspec.centers,
            }
        )
        out = extra.merge(agg, on="bin", how="left")
    else:
        out = agg
    out = out.sort_values("bin").reset_index(drop=True)
    if "n" in out.columns:
        out["n"] = out["n"].fillna(0).astype(int)
    return out


__all__ = [
    "BinSpec",
    "compute_equal_count_bins",
    "assign_bins",
    "summarize_bins",
]


def assign_bins_by_rank(values: Iterable, n_bins: int = 10, *, tie_breaker: str = "first") -> np.ndarray:
    """Assign equal-count bins via deterministic rank mapping on the given values.

    This guarantees (near) equal counts on the same input array (differences at
    most 1 due to rounding), regardless of ties, by using Pandas rank with the
    chosen tie_breaker (default 'first').

    Note: This is intended for binning within a single dataset. For reusing bins
    on new data, prefer compute_equal_count_bins + assign_bins.
    """
    s = _to_series(values)
    s = pd.to_numeric(s, errors="coerce")
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    n = len(s)
    if n == 0:
        return np.array([], dtype=int)
    # Handle NaNs: return -1 for NaNs; bin others by rank
    mask_nan = s.isna().values
    ranks = s.rank(method=tie_breaker)
    idx = ((ranks - 1) * n_bins / n).astype(int).clip(0, n_bins - 1)
    out = idx.values.astype(int)
    out[mask_nan] = -1
    return out

__all__.append("assign_bins_by_rank")


def equal_count_calibration(
    values: Iterable,
    y: Iterable,
    n_bins: int = 10,
    *,
    tie_breaker: str = "first",
) -> pd.DataFrame:
    """Replicate notebook-style equal-count calibration table (no edges).

    This follows the same logic used in the notebook function you shared:
    - Rank predicted values with the chosen tie_breaker (default 'first')
    - Map ranks to bin indices 0..n_bins-1 to produce near-equal counts
    - Group by bin to compute: mean_p, mean_y, n, p_low, p_high
    - Sort by mean_p and return the calibration table

    Returns a DataFrame with columns:
      ['bin', 'mean_p', 'mean_y', 'n', 'p_low', 'p_high']
    """
    p = pd.to_numeric(_to_series(values), errors="coerce")
    yy = pd.to_numeric(_to_series(y), errors="coerce")
    df = pd.DataFrame({"p": p, "y": yy}).dropna()
    if len(df) == 0:
        raise ValueError("No valid rows after dropping NaNs in values/y")
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    n = len(df)
    ranks = df["p"].rank(method=tie_breaker)
    bin_idx = ((ranks - 1) * n_bins / n).astype(int).clip(0, n_bins - 1)
    df["bin"] = bin_idx

    cal = (
        df.groupby("bin", observed=True)
        .agg(
            mean_p=("p", "mean"),
            mean_y=("y", "mean"),
            n=("y", "size"),
            p_low=("p", "min"),
            p_high=("p", "max"),
        )
        .reset_index(drop=False)
    )
    cal = cal.sort_values("mean_p").reset_index(drop=True)
    return cal

__all__.append("equal_count_calibration")
