"""Basic swara analysis helpers built on tonic-normalized pitch contours."""

from __future__ import annotations

import numpy as np


def get_swara_reference_cents() -> dict:
    """Return canonical swara positions in cents relative to Sa."""

    return {
        "Sa": 0.0,
        "re": 100.0,
        "Re": 200.0,
        "ga": 300.0,
        "Ga": 400.0,
        "Ma": 500.0,
        "Ma^": 600.0,
        "Pa": 700.0,
        "dha": 800.0,
        "Dha": 900.0,
        "ni": 1000.0,
        "Ni": 1100.0,
    }


def circular_distance_cents(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute circular distances in cents on a 1200-cent octave.
    values shape: (n,)
    targets shape: (m,)
    returns shape: (n, m)
    """

    values = np.asarray(values, dtype=float).reshape(-1, 1)
    targets = np.asarray(targets, dtype=float).reshape(1, -1)

    diff = np.abs(values - targets)
    return np.minimum(diff, 1200.0 - diff)


def assign_swaras_to_contour(
    normalized_result: dict,
    tolerance_cents: float = 35.0,
) -> dict:
    """
    Assign each voiced folded pitch sample to the nearest swara if it is within tolerance.
    """

    times = normalized_result["times"]
    voiced_mask = normalized_result["voiced_mask"]
    folded_cents_signed = normalized_result["relative_cents_folded"]

    folded_cents = np.full_like(folded_cents_signed, np.nan, dtype=float)
    valid_mask = voiced_mask & np.isfinite(folded_cents_signed)
    folded_cents[valid_mask] = np.mod(folded_cents_signed[valid_mask], 1200.0)

    swara_reference_cents = get_swara_reference_cents()
    swara_names = list(swara_reference_cents.keys())
    swara_targets = np.array([swara_reference_cents[name] for name in swara_names], dtype=float)

    assigned_swaras = np.full(times.shape, None, dtype=object)
    nearest_swara_cents = np.full(times.shape, np.nan, dtype=float)
    distance_to_swara_cents = np.full(times.shape, np.nan, dtype=float)
    confident_mask = np.zeros(times.shape, dtype=bool)

    valid_values = folded_cents[valid_mask]
    if len(valid_values) == 0:
        return {
            "times": times,
            "voiced_mask": voiced_mask,
            "folded_cents": folded_cents,
            "assigned_swaras": assigned_swaras,
            "nearest_swara_cents": nearest_swara_cents,
            "distance_to_swara_cents": distance_to_swara_cents,
            "confident_mask": confident_mask,
            "tolerance_cents": float(tolerance_cents),
            "swara_reference_cents": swara_reference_cents,
        }

    dists = circular_distance_cents(valid_values, swara_targets)
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(len(valid_values)), nearest_idx]
    nearest_names = np.array([swara_names[i] for i in nearest_idx], dtype=object)
    nearest_targets = swara_targets[nearest_idx]

    confident_valid = nearest_dist <= tolerance_cents

    valid_indices = np.where(valid_mask)[0]
    distance_to_swara_cents[valid_indices] = nearest_dist
    nearest_swara_cents[valid_indices] = nearest_targets
    confident_mask[valid_indices] = confident_valid

    assigned_swaras[valid_indices[confident_valid]] = nearest_names[confident_valid]

    return {
        "times": times,
        "voiced_mask": voiced_mask,
        "folded_cents": folded_cents,
        "assigned_swaras": assigned_swaras,
        "nearest_swara_cents": nearest_swara_cents,
        "distance_to_swara_cents": distance_to_swara_cents,
        "confident_mask": confident_mask,
        "tolerance_cents": float(tolerance_cents),
        "swara_reference_cents": swara_reference_cents,
    }


def summarize_swara_distribution(swara_assignment: dict) -> dict:
    """Summarize swara occupancy counts and proportions."""

    assigned_swaras = swara_assignment["assigned_swaras"]
    voiced_mask = swara_assignment["voiced_mask"]
    confident_mask = swara_assignment["confident_mask"]
    swara_reference_cents = swara_assignment["swara_reference_cents"]

    swara_names = list(swara_reference_cents.keys())

    n_voiced_frames = int(np.sum(voiced_mask))
    n_confident_frames = int(np.sum(confident_mask))
    unassigned_frames = int(n_voiced_frames - n_confident_frames)

    swara_counts = {name: 0 for name in swara_names}
    for name in swara_names:
        swara_counts[name] = int(np.sum(assigned_swaras == name))

    if n_confident_frames > 0:
        swara_proportions = {
            name: float(swara_counts[name] / n_confident_frames)
            for name in swara_names
        }
    else:
        swara_proportions = {name: 0.0 for name in swara_names}

    sorted_swaras = sorted(
        swara_names,
        key=lambda name: swara_proportions[name],
        reverse=True,
    )

    dominant_swaras = [name for name in sorted_swaras if swara_counts[name] > 0][:3]
    least_used_swaras = [name for name in sorted_swaras[::-1] if swara_counts[name] > 0][:3]

    confident_ratio = float(n_confident_frames / n_voiced_frames) if n_voiced_frames > 0 else 0.0

    return {
        "n_voiced_frames": n_voiced_frames,
        "n_confident_frames": n_confident_frames,
        "confident_ratio": confident_ratio,
        "swara_counts": swara_counts,
        "swara_proportions": swara_proportions,
        "dominant_swaras": dominant_swaras,
        "least_used_swaras": least_used_swaras,
        "unassigned_frames": unassigned_frames,
    }


def generate_basic_swara_comment(tonic_result: dict, swara_summary: dict) -> str:
    """Generate a short text comment from tonic and swara summaries."""

    track_id = tonic_result.get("track_id", "unknown track")
    tonic_hz = tonic_result.get("tonic_hz")

    if tonic_hz is None:
        return (
            f"For {track_id}, swara analysis could not proceed because tonic estimation "
            f"did not produce a valid Sa."
        )

    dominant_swaras = swara_summary.get("dominant_swaras", [])
    confident_ratio = swara_summary.get("confident_ratio", 0.0)
    n_confident_frames = swara_summary.get("n_confident_frames", 0)

    if len(dominant_swaras) == 0:
        return (
            f"For {track_id}, the estimated tonic is {tonic_hz:.2f} Hz, but no swara "
            f"regions were confidently assigned from the tonic-normalized contour."
        )

    dominant_text = ", ".join(dominant_swaras[:-1]) + f" and {dominant_swaras[-1]}" if len(dominant_swaras) > 1 else dominant_swaras[0]

    return (
        f"For {track_id}, the estimated tonic is {tonic_hz:.2f} Hz. "
        f"After tonic normalization, the contour shows strongest occupancy near "
        f"{dominant_text}. About {100.0 * confident_ratio:.1f}% of voiced frames "
        f"({n_confident_frames} frames) were confidently assigned to canonical swara regions."
    )