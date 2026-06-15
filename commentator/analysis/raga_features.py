"""
Feature extraction helpers for small 3-raga baseline.

Assumes you already have:
- a PitchContour object (times, frequencies, voiced_mask, voiced_frequencies,
  source, track_id),
- the tonic + swara pipeline from tonic_estimator and swara_analyzer.

This module:
- extracts a time segment from a PitchContour,
- runs tonic normalization (optionally reusing track-level tonic),
- builds a 36-dim feature vector per segment:
  24-bin pitch-class histogram (0-1200 cents, 50c bins)
  + 12 swara occupancy proportions.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

from .tonic_estimator import estimate_tonic, normalize_to_tonic
from .swara_analyzer import (
    assign_swaras_to_contour,
    summarize_swara_distribution,
)


def slice_pitch_contour(
    pitch_obj,
    start_time: float,
    end_time: float,
):
    """
    Return a shallow slice of PitchContour between [start_time, end_time].

    This assumes pitch_obj has:
        times: np.ndarray (T,)
        frequencies: np.ndarray (T,)
        voiced_mask: np.ndarray (T,)
        voiced_frequencies: np.ndarray (N_voiced,) or property

    Returns a new object of the same class with arrays restricted
    to the given time range.
    """
    times = pitch_obj.times
    idx = (times >= start_time) & (times < end_time)

    if not np.any(idx):
        # Return an empty-ish contour of the same class
        # (you can customize this if your PitchContour constructor differs)
        new_obj = pitch_obj.__class__(
            times=np.array([], dtype=float),
            frequencies=np.array([], dtype=float),
            voiced_mask=np.array([], dtype=bool),
            source=pitch_obj.source,
            track_id=pitch_obj.track_id,
        )
        return new_obj

    new_times = times[idx]
    new_freqs = pitch_obj.frequencies[idx]
    new_voiced_mask = pitch_obj.voiced_mask[idx]

    # Option 1: recompute voiced_frequencies lazily inside PitchContour
    # If your class stores it explicitly, you can set it here.
    new_obj = pitch_obj.__class__(
        times=new_times,
        frequencies=new_freqs,
        voiced_mask=new_voiced_mask,
        source=pitch_obj.source,
        track_id=pitch_obj.track_id,
    )
    return new_obj


def compute_pitchclass_histogram(
    normalized_result: Dict[str, Any],
    n_bins: int = 24,
) -> np.ndarray:
    """
    Compute tonic-normalized pitch-class histogram over [0, 1200) cents.

    Uses relative_cents_folded from normalize_to_tonic.
    Returns a length-n_bins vector normalized to sum to 1 (if any data).
    """
    folded = normalized_result["relative_cents_folded"]
    voiced_mask = normalized_result["voiced_mask"]

    # Only use voiced and finite values
    valid_mask = voiced_mask & np.isfinite(folded)
    values = folded[valid_mask]

    if len(values) == 0:
        return np.zeros(n_bins, dtype=float)

    values = np.mod(values, 1200.0)
    bin_edges = np.linspace(0.0, 1200.0, n_bins + 1)
    hist, _ = np.histogram(values, bins=bin_edges)
    hist = hist.astype(float)

    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


def compute_swara_proportion_vector(
    swara_summary: Dict[str, Any],
    swara_order=None,
) -> np.ndarray:
    """
    Turn swara_proportions dict into a fixed-order vector.

    swara_order: optional list defining the order of swaras.
                 If None, uses the sorted keys of swara_proportions.

    Returns a 1D float array.
    """
    swara_proportions = swara_summary["swara_proportions"]

    if swara_order is None:
        swara_order = sorted(swara_proportions.keys())

    vec = np.array([float(swara_proportions.get(name, 0.0)) for name in swara_order],
                   dtype=float)
    return vec


def extract_segment_features(
    pitch_obj,
    start_time: float,
    end_time: float,
    *,
    ref_hz: float = 55.0,
    # tonic estimation parameters can be passed or you can reuse track-level tonic
    bin_size_cents: float = 10.0,
    min_hz: float = 60.0,
    max_hz: float = 1000.0,
    smooth_kernel_size: int = 7,
    candidate_min_hz: float = 80.0,
    candidate_max_hz: float = 400.0,
    swara_tolerance_cents: float = 35.0,
    n_pc_bins: int = 24,
    swara_order=None,
    reuse_tonic_result: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract a 36-dim feature vector for a given time segment of a track.

    Pipeline per segment:
        1. Slice PitchContour to [start_time, end_time].
        2. Estimate tonic (or reuse track-level tonic_result).
        3. Normalize to tonic.
        4. Assign swaras, summarize distribution.
        5. Build feature vector = pitch-class hist + swara proportions.

    Returns
    -------
    features : np.ndarray
        1D vector of length n_pc_bins + n_swaras.
    meta : dict
        {
            "track_id": str,
            "start_time": float,
            "end_time": float,
            "tonic_result": dict,
            "swara_summary": dict,
        }
    """
    segment_pitch = slice_pitch_contour(pitch_obj, start_time, end_time)

    # If no frames in segment, return zeros
    if segment_pitch.times.size == 0:
        # We still want to know the dimensionality of swara vector.
        # Use a dummy run on the full track if needed, but for now we assume 12.
        dummy_swara_dim = 12 if swara_order is None else len(swara_order)
        feat = np.zeros(n_pc_bins + dummy_swara_dim, dtype=float)
        meta = {
            "track_id": getattr(pitch_obj, "track_id", None),
            "start_time": start_time,
            "end_time": end_time,
            "tonic_result": None,
            "swara_summary": None,
        }
        return feat, meta

    # 1. Tonic: either reuse supplied tonic_result, or estimate from segment
    if reuse_tonic_result is not None:
        tonic_result = reuse_tonic_result
    else:
        tonic_result = estimate_tonic(
            segment_pitch,
            ref_hz=ref_hz,
            bin_size_cents=bin_size_cents,
            min_hz=min_hz,
            max_hz=max_hz,
            smooth_kernel_size=smooth_kernel_size,
            candidate_min_hz=candidate_min_hz,
            candidate_max_hz=candidate_max_hz,
        )

    # 2. Normalize to tonic
    normalized_result = normalize_to_tonic(
        segment_pitch,
        tonic_result=tonic_result,
        ref_hz=ref_hz,
    )

    # 3. Swara assignment + summary
    swara_assignment = assign_swaras_to_contour(
        normalized_result,
        tolerance_cents=swara_tolerance_cents,
    )
    swara_summary = summarize_swara_distribution(swara_assignment)

    # 4. Features: pitch-class histogram + swara proportions
    pc_hist = compute_pitchclass_histogram(
        normalized_result,
        n_bins=n_pc_bins,
    )

    swara_vec = compute_swara_proportion_vector(
        swara_summary,
        swara_order=swara_order,
    )

    features = np.concatenate([pc_hist, swara_vec], axis=0)

    meta = {
        "track_id": getattr(pitch_obj, "track_id", None),
        "start_time": start_time,
        "end_time": end_time,
        "tonic_result": tonic_result,
        "swara_summary": swara_summary,
    }

    return features, meta