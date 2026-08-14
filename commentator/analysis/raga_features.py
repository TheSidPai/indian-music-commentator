"""
Stage-2 feature extraction from Stage-1 schema.

Purpose
-------
Convert the structured Stage-1 musical analysis output into a fixed-length
numeric feature vector suitable for raga classification and related MIR tasks.

Input
-----
A Stage-1 schema dict produced by build_stage1_schema(...)

Output
------
- feature_vector: np.ndarray of shape (n_features,)
- feature_names: list[str]
- meta: dict with track metadata
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np


CANONICAL_SWARA_ORDER = [
    "Sa", "re", "Re", "ga", "Ga", "Ma", "Ma^", "Pa", "dha", "Dha", "ni", "Ni"
]

DROP_NAMES = {
    "tonic_pc_cents",
    "n_voiced_used",
    "log_n_voiced_used",
    "min_relative_cents",
    "max_relative_cents",
    "median_relative_cents",
    "hist_bin_size_cents",
    "hist_peak_2_cents",
    "hist_peak_2_height",
    "hist_peak_3_cents",
    "hist_peak_3_height",
}
def apply_feature_subset(x, names):
    mask = [name not in DROP_NAMES for name in names]
    x_sub = x[mask]
    names_sub = [n for n, keep in zip(names, mask) if keep]
    return x_sub, names_sub

def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_array(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _compute_histogram_entropy(hist: np.ndarray) -> float:
    hist = np.asarray(hist, dtype=float)
    total = np.sum(hist)
    if total <= 0:
        return 0.0
    p = hist / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def _compute_histogram_concentration(hist: np.ndarray) -> float:
    hist = np.asarray(hist, dtype=float)
    total = np.sum(hist)
    if total <= 0:
        return 0.0
    p = hist / total
    return float(np.sum(p ** 2))


def _get_peak_indices(values: np.ndarray, k: int = 3, exclusion_radius: int = 2) -> List[int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []

    working = values.copy()
    peak_indices: List[int] = []

    for _ in range(k):
        idx = int(np.argmax(working))
        if not np.isfinite(working[idx]) or working[idx] <= -np.inf / 2:
            break

        peak_indices.append(idx)

        left = max(0, idx - exclusion_radius)
        right = min(len(working), idx + exclusion_radius + 1)
        working[left:right] = -np.inf

    return peak_indices


def _extract_histogram_peak_features(
    pitch_histogram: Dict[str, Any],
    n_peaks: int = 3,
) -> Tuple[List[float], List[str]]:
    bin_centers = _safe_array(pitch_histogram.get("bin_centers"))
    hist = _safe_array(pitch_histogram.get("hist"))
    hist_smoothed = _safe_array(pitch_histogram.get("hist_smoothed"))

    feature_values: List[float] = []
    feature_names: List[str] = []

    if hist_smoothed.size == 0 or bin_centers.size == 0:
        for i in range(1, n_peaks + 1):
            feature_values.extend([0.0, 0.0])
            feature_names.extend([
                f"hist_peak_{i}_cents",
                f"hist_peak_{i}_height",
            ])
        feature_values.extend([0.0, 0.0])
        feature_names.extend(["hist_entropy", "hist_concentration"])
        return feature_values, feature_names

    peak_indices = _get_peak_indices(hist_smoothed, k=n_peaks, exclusion_radius=2)

    total_smoothed = float(np.sum(hist_smoothed))
    if total_smoothed <= 0:
        total_smoothed = 1.0

    for i in range(n_peaks):
        if i < len(peak_indices):
            idx = peak_indices[i]
            peak_cents = float(bin_centers[idx])
            peak_height = float(hist_smoothed[idx] / total_smoothed)
        else:
            peak_cents = 0.0
            peak_height = 0.0

        feature_values.extend([peak_cents, peak_height])
        feature_names.extend([
            f"hist_peak_{i + 1}_cents",
            f"hist_peak_{i + 1}_height",
        ])

    feature_values.append(_compute_histogram_entropy(hist_smoothed))
    feature_names.append("hist_entropy")

    feature_values.append(_compute_histogram_concentration(hist_smoothed))
    feature_names.append("hist_concentration")

    return feature_values, feature_names


def _extract_tonic_features(stage1: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    tonic = stage1.get("tonic", {})

    tonic_hz = _safe_float(tonic.get("tonic_hz"))
    tonic_pc_cents = _safe_float(tonic.get("tonic_pc_cents"))
    n_voiced_used = _safe_float(tonic.get("n_voiced_used"))
    target_hz = _safe_float(tonic.get("target_hz"))

    candidate_hz = tonic.get("candidate_hz", [])
    candidate_count = float(len(candidate_hz)) if candidate_hz is not None else 0.0

    values = [
        tonic_hz,
        float(np.log1p(max(0.0, tonic_hz))),
        tonic_pc_cents,
        n_voiced_used,
        float(np.log1p(max(0.0, n_voiced_used))),
        target_hz,
        candidate_count,
    ]
    names = [
        "tonic_hz",
        "log_tonic_hz",
        "tonic_pc_cents",
        "n_voiced_used",
        "log_n_voiced_used",
        "target_hz",
        "n_tonic_candidates",
    ]
    return values, names


def _extract_swara_features(
    stage1: Dict[str, Any],
    swara_order: List[str] | None = None,
) -> Tuple[List[float], List[str]]:
    if swara_order is None:
        swara_order = CANONICAL_SWARA_ORDER

    swara = stage1.get("swara", {})
    swara_proportions = swara.get("swara_proportions", {})
    swara_counts = swara.get("swara_counts", {})

    values: List[float] = []
    names: List[str] = []

    for sw in swara_order:
        values.append(_safe_float(swara_proportions.get(sw), default=0.0))
        names.append(f"swara_prop_{sw}")

    for sw in swara_order:
        count_val = _safe_float(swara_counts.get(sw), default=0.0)
        values.append(float(np.log1p(max(0.0, count_val))))
        names.append(f"log_swara_count_{sw}")

    n_voiced_frames = _safe_float(swara.get("n_voiced_frames"))
    n_confident_frames = _safe_float(swara.get("n_confident_frames"))
    confident_ratio = _safe_float(swara.get("confident_ratio"))
    unassigned_frames = _safe_float(swara.get("unassigned_frames"))

    values.extend([
        n_voiced_frames,
        float(np.log1p(max(0.0, n_voiced_frames))),
        n_confident_frames,
        float(np.log1p(max(0.0, n_confident_frames))),
        confident_ratio,
        unassigned_frames,
        float(np.log1p(max(0.0, unassigned_frames))),
    ])
    names.extend([
        "n_voiced_frames",
        "log_n_voiced_frames",
        "n_confident_frames",
        "log_n_confident_frames",
        "confident_ratio",
        "unassigned_frames",
        "log_unassigned_frames",
    ])

    return values, names


def _extract_range_features(stage1: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    pitch_summary = stage1.get("pitch_summary", {})
    voiced_range = pitch_summary.get("voiced_range_cents", {})

    min_cents = _safe_float(voiced_range.get("min_cents"))
    max_cents = _safe_float(voiced_range.get("max_cents"))
    median_cents = _safe_float(voiced_range.get("median_cents"))
    span_cents = max_cents - min_cents

    values = [
        min_cents,
        max_cents,
        median_cents,
        span_cents,
    ]
    names = [
        "min_relative_cents",
        "max_relative_cents",
        "median_relative_cents",
        "range_span_cents",
    ]
    return values, names


def _extract_pitch_histogram_features(stage1: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    pitch_summary = stage1.get("pitch_summary", {})
    pitch_histogram = pitch_summary.get("pitch_histogram", {})

    bin_size_cents = _safe_float(pitch_histogram.get("bin_size_cents"))
    ref_hz = _safe_float(pitch_histogram.get("ref_hz"))

    peak_values, peak_names = _extract_histogram_peak_features(pitch_histogram, n_peaks=3)

    values = [bin_size_cents, ref_hz] + peak_values
    names = ["hist_bin_size_cents", "hist_ref_hz"] + peak_names
    return values, names

def _get_relative_pitch_track(stage1: Dict[str, Any]) -> np.ndarray:
    pitch_summary = stage1.get("pitch_summary", {})
    pitch_histogram = pitch_summary.get("pitch_histogram", {})

    rel_cents = None

    # Prefer an explicit relative contour if present
    if "relative_cents" in pitch_summary:
        rel_cents = pitch_summary.get("relative_cents")
    elif "relative_cents" in pitch_histogram:
        rel_cents = pitch_histogram.get("relative_cents")
    elif "voiced_relative_cents" in pitch_summary:
        rel_cents = pitch_summary.get("voiced_relative_cents")

    arr = _safe_array(rel_cents)
    arr = arr[np.isfinite(arr)]
    return arr


def _quantize_relative_cents_to_swara(
    rel_cents: np.ndarray,
    swara_order: List[str] | None = None,
) -> List[str]:
    if swara_order is None:
        swara_order = CANONICAL_SWARA_ORDER

    swara_centers = np.array(
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
        dtype=float,
    )

    rel_cents = np.asarray(rel_cents, dtype=float)
    rel_cents = rel_cents[np.isfinite(rel_cents)]

    if rel_cents.size == 0:
        return []

    rel_mod = np.mod(rel_cents, 1200.0)

    seq = []
    for x in rel_mod:
        idx = int(np.argmin(np.abs(swara_centers - x)))
        seq.append(swara_order[idx])
    return seq


def _find_stable_regions(
    rel_cents: np.ndarray,
    slope_threshold_cents: float = 8.0,
    min_region_len: int = 5,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    rel_cents = np.asarray(rel_cents, dtype=float)
    rel_cents = rel_cents[np.isfinite(rel_cents)]

    if rel_cents.size < 2:
        return [], []

    diff = np.diff(rel_cents)
    stable_mask = np.abs(diff) <= slope_threshold_cents

    stable_regions: List[Tuple[int, int]] = []
    transition_regions: List[Tuple[int, int]] = []

    start = 0
    current_stable = bool(stable_mask[0])

    for i in range(1, len(stable_mask)):
        state = bool(stable_mask[i])
        if state != current_stable:
            end = i
            if current_stable:
                if (end - start + 1) >= min_region_len:
                    stable_regions.append((start, end + 1))
            else:
                transition_regions.append((start, end + 1))
            start = i
            current_stable = state

    end = len(stable_mask)
    if current_stable:
        if (end - start + 1) >= min_region_len:
            stable_regions.append((start, end + 1))
    else:
        transition_regions.append((start, end + 1))

    return stable_regions, transition_regions

def _extract_contour_movement_features(stage1: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    rel_cents = _get_relative_pitch_track(stage1)

    if rel_cents.size < 2:
        values = [0.0] * 8
        names = [
            "mean_abs_pitch_diff_cents",
            "std_pitch_diff_cents",
            "mean_positive_pitch_diff_cents",
            "mean_negative_pitch_diff_cents",
            "frac_rising_frames",
            "frac_falling_frames",
            "frac_flat_frames",
            "mean_pitch_step_size_cents",
        ]
        return values, names

    diff = np.diff(rel_cents)
    abs_diff = np.abs(diff)

    pos = diff[diff > 0]
    neg = diff[diff < 0]
    flat_frac = float(np.mean(abs_diff <= 8.0))
    rising_frac = float(np.mean(diff > 8.0))
    falling_frac = float(np.mean(diff < -8.0))

    values = [
        float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        float(np.std(diff)) if diff.size else 0.0,
        float(np.mean(pos)) if pos.size else 0.0,
        float(np.mean(np.abs(neg))) if neg.size else 0.0,
        rising_frac,
        falling_frac,
        flat_frac,
        float(np.mean(abs_diff[abs_diff > 8.0])) if np.any(abs_diff > 8.0) else 0.0,
    ]

    names = [
        "mean_abs_pitch_diff_cents",
        "std_pitch_diff_cents",
        "mean_positive_pitch_diff_cents",
        "mean_negative_pitch_diff_cents",
        "frac_rising_frames",
        "frac_falling_frames",
        "frac_flat_frames",
        "mean_pitch_step_size_cents",
    ]
    return values, names

def _extract_stability_transition_features(stage1: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    rel_cents = _get_relative_pitch_track(stage1)

    if rel_cents.size < 2:
        values = [0.0] * 8
        names = [
            "n_stable_regions",
            "mean_stable_region_len",
            "max_stable_region_len",
            "stable_frame_ratio",
            "n_transition_regions",
            "mean_transition_region_len",
            "max_transition_region_len",
            "transition_frame_ratio",
        ]
        return values, names

    stable_regions, transition_regions = _find_stable_regions(
        rel_cents,
        slope_threshold_cents=8.0,
        min_region_len=5,
    )

    stable_lengths = np.array([e - s for s, e in stable_regions], dtype=float)
    transition_lengths = np.array([e - s for s, e in transition_regions], dtype=float)

    total_frames = float(len(rel_cents))
    stable_frames = float(np.sum(stable_lengths)) if stable_lengths.size else 0.0
    transition_frames = float(np.sum(transition_lengths)) if transition_lengths.size else 0.0

    values = [
        float(len(stable_regions)),
        float(np.mean(stable_lengths)) if stable_lengths.size else 0.0,
        float(np.max(stable_lengths)) if stable_lengths.size else 0.0,
        stable_frames / total_frames if total_frames > 0 else 0.0,
        float(len(transition_regions)),
        float(np.mean(transition_lengths)) if transition_lengths.size else 0.0,
        float(np.max(transition_lengths)) if transition_lengths.size else 0.0,
        transition_frames / total_frames if total_frames > 0 else 0.0,
    ]

    names = [
        "n_stable_regions",
        "mean_stable_region_len",
        "max_stable_region_len",
        "stable_frame_ratio",
        "n_transition_regions",
        "mean_transition_region_len",
        "max_transition_region_len",
        "transition_frame_ratio",
    ]
    return values, names

def _extract_swara_pattern_features(
    stage1: Dict[str, Any],
    swara_order: List[str] | None = None,
) -> Tuple[List[float], List[str]]:
    if swara_order is None:
        swara_order = CANONICAL_SWARA_ORDER

    rel_cents = _get_relative_pitch_track(stage1)
    swara_seq = _quantize_relative_cents_to_swara(rel_cents, swara_order=swara_order)

    selected_bigrams = [
        ("Sa", "Re"),
        ("Re", "Ga"),
        ("Ga", "Ma"),
        ("Ma", "Pa"),
        ("Pa", "Dha"),
        ("Dha", "Ni"),
        ("Ni", "Sa"),
        ("Sa", "Ni"),
    ]

    selected_trigrams = [
        ("Sa", "Re", "Ga"),
        ("Ga", "Ma", "Pa"),
        ("Pa", "Dha", "Ni"),
        ("Ni", "Dha", "Pa"),
        ("Ga", "Re", "Sa"),
        ("Ni", "Sa", "Re"),
    ]

    bigrams = list(zip(swara_seq[:-1], swara_seq[1:])) if len(swara_seq) >= 2 else []
    trigrams = list(zip(swara_seq[:-2], swara_seq[1:-1], swara_seq[2:])) if len(swara_seq) >= 3 else []

    n_big = max(1, len(bigrams))
    n_tri = max(1, len(trigrams))

    values: List[float] = []
    names: List[str] = []

    for bg in selected_bigrams:
        count = sum(1 for x in bigrams if x == bg)
        values.append(float(count) / n_big)
        names.append(f"bigram_prop_{bg[0]}_{bg[1]}")

    for tg in selected_trigrams:
        count = sum(1 for x in trigrams if x == tg)
        values.append(float(count) / n_tri)
        names.append(f"trigram_prop_{tg[0]}_{tg[1]}_{tg[2]}")

    return values, names

def extract_raga_features_from_stage1(
    stage1: Dict[str, Any],
    swara_order: List[str] | None = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Convert a Stage-1 schema dict into a fixed-length numeric feature vector.

    Parameters
    ----------
    stage1 : dict
        Output of build_stage1_schema(...)
    swara_order : list[str] | None
        Fixed swara ordering. Defaults to canonical order.

    Returns
    -------
    feature_vector : np.ndarray
        1D float vector.
    feature_names : list[str]
        Names aligned with the feature vector.
    meta : dict
        Lightweight metadata for traceability.
    """
    feature_values: List[float] = []
    feature_names: List[str] = []

    tonic_values, tonic_names = _extract_tonic_features(stage1)
    feature_values.extend(tonic_values)
    feature_names.extend(tonic_names)

    swara_values, swara_names = _extract_swara_features(stage1, swara_order=swara_order)
    feature_values.extend(swara_values)
    feature_names.extend(swara_names)

    range_values, range_names = _extract_range_features(stage1)
    feature_values.extend(range_values)
    feature_names.extend(range_names)

    hist_values, hist_names = _extract_pitch_histogram_features(stage1)
    feature_values.extend(hist_values)
    feature_names.extend(hist_names)

    movement_values, movement_names = _extract_contour_movement_features(stage1)
    feature_values.extend(movement_values)
    feature_names.extend(movement_names)

    stability_values, stability_names = _extract_stability_transition_features(stage1)
    feature_values.extend(stability_values)
    feature_names.extend(stability_names)

    pattern_values, pattern_names = _extract_swara_pattern_features(stage1, swara_order=swara_order)
    feature_values.extend(pattern_values)
    feature_names.extend(pattern_names)

    feature_vector = np.asarray(feature_values, dtype=float)

    feature_vector, feature_names = apply_feature_subset(feature_vector, feature_names)

    meta = {
        "track_id": stage1.get("meta", {}).get("track_id"),
        "raga_label": stage1.get("meta", {}).get("raga_label"),
        "source": stage1.get("meta", {}).get("source"),
        "analysis_version": stage1.get("meta", {}).get("analysis_version"),
        "n_features": int(feature_vector.shape[0]),
    }

    # for i, name in enumerate(feature_names):
    #     print(i, name)

    return feature_vector, feature_names, meta