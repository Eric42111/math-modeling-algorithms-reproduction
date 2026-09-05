"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
Zero-dependency implementation using only Python standard library.

Supports four indicator types (benefit / cost / optimal-value / interval),
vector normalization, subjective weighting, closeness-coefficient ranking,
and weight sensitivity analysis.

Usage:
    py topsis.py
"""

import math


# ---------------------------------------------------------------------------
# Indicator transformation
# ---------------------------------------------------------------------------

def positive_transform(values, kind, best=None, low=None, high=None,
                       mode="range"):
    """Convert an indicator column to benefit-type (larger is better).

    Parameters
    ----------
    values : list[float]
    kind   : 'max' | 'min' | 'mid' | 'interval'
    best   : target value, required when kind='mid'
    low, high : interval bounds, required when kind='interval'
    mode   : 'range' (max - x) or 'reciprocal' (1 / x), for kind='min'
    """
    if kind == "max":
        return [float(v) for v in values]

    if kind == "min":
        if mode == "reciprocal":
            if min(values) <= 0:
                raise ValueError(
                    "reciprocal mode requires all values > 0; "
                    "use mode='range' instead")
            return [1.0 / v for v in values]
        m = max(values)
        return [m - v for v in values]

    if kind == "mid":
        if best is None:
            raise ValueError("kind='mid' requires the 'best' parameter")
        dev = max(abs(v - best) for v in values)
        if dev == 0:
            return [1.0] * len(values)
        return [1.0 - abs(v - best) / dev for v in values]

    if kind == "interval":
        if low is None or high is None:
            raise ValueError(
                "kind='interval' requires 'low' and 'high' parameters")
        M = max(low - min(values), max(values) - high)
        if M <= 0:
            return [1.0] * len(values)
        out = []
        for v in values:
            if v < low:
                out.append(1.0 - (low - v) / M)
            elif v > high:
                out.append(1.0 - (v - high) / M)
            else:
                out.append(1.0)
        return out

    raise ValueError(f"unknown indicator type: {kind}")


# ---------------------------------------------------------------------------
# Normalization & weighting
# ---------------------------------------------------------------------------

def normalize(matrix):
    """Vector normalization: z_ij = x_ij / sqrt(sum_i x_ij^2)."""
    n_col = len(matrix[0])
    norms = []
    for j in range(n_col):
        s = math.sqrt(sum(row[j] ** 2 for row in matrix))
        norms.append(s if s > 0 else 1e-12)
    return [[row[j] / norms[j] for j in range(n_col)] for row in matrix]


def apply_weights(matrix, weights):
    """Weighted normalized matrix: v_ij = w_j * z_ij."""
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {sum(weights):.6f}")
    n_col = len(matrix[0])
    if len(weights) != n_col:
        raise ValueError(
            f"weight count {len(weights)} != indicator count {n_col}")
    return [[row[j] * weights[j] for j in range(n_col)] for row in matrix]


# ---------------------------------------------------------------------------
# Core TOPSIS
# ---------------------------------------------------------------------------

def _euclid(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def topsis(raw_matrix, kinds, weights, **kw):
    """Run the full TOPSIS pipeline.

    Parameters
    ----------
    raw_matrix : list[list[float]]  -- rows = alternatives, cols = indicators
    kinds      : list[str]          -- indicator type per column
    weights    : list[float]        -- must sum to 1

    Returns
    -------
    dict with keys:
        C        : list[float]  closeness coefficients (higher is better)
        D_pos    : list[float]  distances to positive-ideal solution
        D_neg    : list[float]  distances to negative-ideal solution
        rank     : list[int]    alternative indices sorted by C (desc)
        weighted : list[list[float]]  weighted normalized matrix
        v_pos    : list[float]  positive-ideal solution
        v_neg    : list[float]  negative-ideal solution
    """
    n_row, n_col = len(raw_matrix), len(raw_matrix[0])

    # Step 1: transform all indicators to benefit-type
    cols = []
    for j in range(n_col):
        col = [raw_matrix[i][j] for i in range(n_row)]
        cols.append(positive_transform(col, kinds[j], **kw))
    positive = [[cols[j][i] for j in range(n_col)] for i in range(n_row)]

    # Step 2: vector normalization
    normalized = normalize(positive)

    # Step 3: apply weights
    weighted = apply_weights(normalized, weights)

    # Step 4: ideal solutions
    v_pos = [max(row[j] for row in weighted) for j in range(n_col)]
    v_neg = [min(row[j] for row in weighted) for j in range(n_col)]

    # Step 5: distances
    d_pos = [_euclid(weighted[i], v_pos) for i in range(n_row)]
    d_neg = [_euclid(weighted[i], v_neg) for i in range(n_row)]

    # Step 6: closeness coefficient
    scores = []
    for i in range(n_row):
        s = d_pos[i] + d_neg[i]
        scores.append(d_neg[i] / s if s > 0 else 0.0)

    # Step 7: rank (descending C; ties broken by ascending D+)
    rank = sorted(range(n_row), key=lambda i: (-scores[i], d_pos[i]))

    return {
        "C": scores, "D_pos": d_pos, "D_neg": d_neg, "rank": rank,
        "weighted": weighted, "v_pos": v_pos, "v_neg": v_neg,
    }


def sensitivity(raw_matrix, kinds, weight_sets, labels, **kw):
    """Run TOPSIS under multiple weight schemes to test robustness.

    Returns list of (label, weights, rank, scores) tuples.
    """
    results = []
    for w, lab in zip(weight_sets, labels):
        r = topsis(raw_matrix, kinds, w, **kw)
        results.append((lab, w, r["rank"], r["C"]))
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Scenario: optimal NIPT timing for a BMI group (2025 CUMCM Problem C).
    # Four conflicting indicators; data is synthetic for demonstration only.
    plans = ["12w", "14w", "16w", "18w", "20w", "22w"]
    raw = [
        [62.0, 18.0, 15.0, 1.0],
        [74.0, 12.0, 13.0, 1.4],
        [83.0,  8.0, 11.0, 2.2],
        [89.0,  5.0,  9.0, 3.6],
        [93.0,  3.5,  7.0, 5.5],
        [96.0,  2.5,  5.0, 8.0],
    ]
    kinds = ["max", "min", "max", "min"]
    weights = [0.30, 0.20, 0.30, 0.20]

    result = topsis(raw, kinds, weights)
    C, rank = result["C"], result["rank"]

    print(f"{'Rank':<6}{'Plan':<8}{'D+':>10}{'D-':>10}{'C':>10}")
    print("-" * 44)
    for pos, i in enumerate(rank, 1):
        print(f"{pos:<6}{plans[i]:<8}"
              f"{result['D_pos'][i]:>10.4f}"
              f"{result['D_neg'][i]:>10.4f}"
              f"{C[i]:>10.4f}")
    print(f"\nBest: {plans[rank[0]]}  (C = {C[rank[0]]:.4f})")

    # Sensitivity analysis
    weight_sets = [
        [0.30, 0.20, 0.30, 0.20],
        [0.50, 0.20, 0.20, 0.10],
        [0.15, 0.10, 0.45, 0.30],
        [0.25, 0.25, 0.25, 0.25],
    ]
    labels = ["balanced", "accuracy-focused",
              "early-detection-focused", "equal"]

    print(f"\n{'Scheme':<26}{'1st':<8}{'2nd':<8}{'3rd':<8}")
    print("-" * 50)
    for lab, w, rk, sc in sensitivity(raw, kinds, weight_sets, labels):
        print(f"{lab:<26}{plans[rk[0]]:<8}{plans[rk[1]]:<8}{plans[rk[2]]:<8}")
