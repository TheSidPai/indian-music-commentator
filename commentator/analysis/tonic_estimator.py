"""Tonic estimation and tonic-normalization helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import PitchContour

import matplotlib.pyplot as plt


def get_clean_voiced_frequencies(
    pitch_obj: PitchContour,
    min_hz: float = 60.0,
    max_hz: float = 1000.0,
) -> np.ndarray:
    """Return voiced frequencies filtered to a usable analysis range."""

    freqs = pitch_obj.voiced_frequencies
    if len(freqs) == 0:
        return np.array([], dtype=float)

    mask = np.isfinite(freqs) & (freqs >= min_hz) & (freqs <= max_hz)
    return freqs[mask]


def hz_to_cents(freqs_hz: np.ndarray, ref_hz: float = 55.0) -> np.ndarray:
    """Convert frequencies in Hz to cents relative to a reference pitch."""

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    return 1200.0 * np.log2(freqs_hz / ref_hz)


def fold_cents_to_octave(cents: np.ndarray) -> np.ndarray:
    """Fold cents values into a single octave [0, 1200)."""

    cents = np.asarray(cents, dtype=float)
    return np.mod(cents, 1200.0)


def smooth_histogram(hist: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Smooth a circular histogram with a wrap-padded moving average."""

    hist = np.asarray(hist, dtype=float)

    if kernel_size <= 1:
        return hist.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(hist, (pad, pad), mode="wrap")
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    return np.convolve(padded, kernel, mode="valid")


def compute_folded_pitch_histogram(
    pitch_obj: PitchContour,
    ref_hz: float = 55.0,
    bin_size_cents: float = 10.0,
    min_hz: float = 60.0,
    max_hz: float = 1000.0,
    smooth_kernel_size: int = 7,
) -> dict:
    """Build a folded pitch-class histogram from voiced frequencies."""

    freqs_hz = get_clean_voiced_frequencies(pitch_obj, min_hz=min_hz, max_hz=max_hz)

    if len(freqs_hz) == 0:
        return {
            "freqs_hz": freqs_hz,
            "cents": np.array([], dtype=float),
            "folded_cents": np.array([], dtype=float),
            "bin_edges": np.array([], dtype=float),
            "bin_centers": np.array([], dtype=float),
            "hist": np.array([], dtype=float),
            "hist_smoothed": np.array([], dtype=float),
            "bin_size_cents": bin_size_cents,
            "ref_hz": ref_hz,
        }

    cents = hz_to_cents(freqs_hz, ref_hz=ref_hz)
    folded_cents = fold_cents_to_octave(cents)

    bin_edges = np.arange(0.0, 1200.0 + bin_size_cents, bin_size_cents)
    hist, _ = np.histogram(folded_cents, bins=bin_edges)
    hist = hist.astype(float)
    hist_smoothed = smooth_histogram(hist, kernel_size=smooth_kernel_size)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "freqs_hz": freqs_hz,
        "cents": cents,
        "folded_cents": folded_cents,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "hist": hist,
        "hist_smoothed": hist_smoothed,
        "bin_size_cents": bin_size_cents,
        "ref_hz": ref_hz,
    }


def select_tonic_pitch_class(hist_data: dict) -> Optional[float]:
    """Select the folded pitch-class center corresponding to the histogram peak."""

    hist_smoothed = hist_data["hist_smoothed"]
    bin_centers = hist_data["bin_centers"]

    if len(hist_smoothed) == 0:
        return None

    peak_idx = int(np.argmax(hist_smoothed))
    return float(bin_centers[peak_idx])


def cents_to_hz(cents: np.ndarray, ref_hz: float = 55.0) -> np.ndarray:
    """Convert cents relative to a reference pitch into Hz."""

    cents = np.asarray(cents, dtype=float)
    return ref_hz * (2.0 ** (cents / 1200.0))


def resolve_tonic_octave(
    freqs_hz: np.ndarray,
    tonic_pc_cents: float,
    ref_hz: float = 55.0,
    candidate_min_hz: float = 80.0,
    candidate_max_hz: float = 400.0,
) -> dict:
    """Choose the tonic octave candidate closest to the target region."""

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if len(freqs_hz) == 0:
        return {"tonic_hz": None, "target_hz": None, "candidate_hz": []}

    low = float(np.percentile(freqs_hz, 10))
    high = float(np.percentile(freqs_hz, 90))
    target_hz = float(np.percentile(freqs_hz, 30))

    octave_offsets = np.arange(-4, 5) * 1200.0
    candidate_cents = tonic_pc_cents + octave_offsets
    candidate_hz = cents_to_hz(candidate_cents, ref_hz=ref_hz)
    candidate_hz = candidate_hz[(candidate_hz >= candidate_min_hz) & (candidate_hz <= candidate_max_hz)]

    if len(candidate_hz) == 0:
        return {"tonic_hz": None, "target_hz": target_hz, "candidate_hz": []}

    in_range_candidates = candidate_hz[(candidate_hz >= low * 0.8) & (candidate_hz <= high * 1.05)]
    if len(in_range_candidates) == 0:
        chosen = float(candidate_hz[np.argmin(np.abs(candidate_hz - target_hz))])
    else:
        chosen = float(in_range_candidates[np.argmin(np.abs(in_range_candidates - target_hz))])

    return {"tonic_hz": chosen, "target_hz": target_hz, "candidate_hz": candidate_hz.tolist()}


def estimate_tonic(
    pitch_obj: PitchContour,
    ref_hz: float = 55.0,
    bin_size_cents: float = 10.0,
    min_hz: float = 60.0,
    max_hz: float = 1000.0,
    smooth_kernel_size: int = 7,
    candidate_min_hz: float = 80.0,
    candidate_max_hz: float = 400.0,
) -> dict:
    """Estimate tonic from a pitch contour using a folded histogram baseline."""

    hist_data = compute_folded_pitch_histogram(
        pitch_obj,
        ref_hz=ref_hz,
        bin_size_cents=bin_size_cents,
        min_hz=min_hz,
        max_hz=max_hz,
        smooth_kernel_size=smooth_kernel_size,
    )
    tonic_pc_cents = select_tonic_pitch_class(hist_data)

    if tonic_pc_cents is None:
        return {
            "source": pitch_obj.source,
            "track_id": pitch_obj.track_id,
            "method": "folded_histogram_v1",
            "tonic_hz": None,
            "tonic_pc_cents": None,
            "n_voiced_used": 0,
            "candidate_hz": [],
            "target_hz": None,
            "hist_data": hist_data,
        }

    octave_data = resolve_tonic_octave(
        hist_data["freqs_hz"],
        tonic_pc_cents=tonic_pc_cents,
        ref_hz=ref_hz,
        candidate_min_hz=candidate_min_hz,
        candidate_max_hz=candidate_max_hz,
    )

    return {
        "source": pitch_obj.source,
        "track_id": pitch_obj.track_id,
        "method": "folded_histogram_v1",
        "tonic_hz": octave_data["tonic_hz"],
        "tonic_pc_cents": float(tonic_pc_cents),
        "n_voiced_used": int(len(hist_data["freqs_hz"])),
        "candidate_hz": octave_data["candidate_hz"],
        "target_hz": octave_data["target_hz"],
        "hist_data": hist_data,
    }


def normalize_to_tonic(
    pitch_obj: PitchContour,
    tonic_result: dict,
    ref_hz: float = 55.0,
) -> dict:
    """Normalize a contour to tonic-centered cents and fold into one octave."""

    tonic_hz = tonic_result.get("tonic_hz")
    times = pitch_obj.times.copy()
    voiced_mask = pitch_obj.voiced_mask.copy()

    cents = np.full_like(pitch_obj.frequencies, np.nan, dtype=float)
    relative_cents = np.full_like(pitch_obj.frequencies, np.nan, dtype=float)
    relative_cents_folded = np.full_like(pitch_obj.frequencies, np.nan, dtype=float)

    valid_mask = voiced_mask & np.isfinite(pitch_obj.frequencies) & (pitch_obj.frequencies > 0)
    if tonic_hz is None:
        return {
            "track_id": pitch_obj.track_id,
            "tonic_hz": None,
            "times": times,
            "voiced_mask": voiced_mask,
            "cents": cents,
            "relative_cents": relative_cents,
            "relative_cents_folded": relative_cents_folded,
        }

    cents[valid_mask] = hz_to_cents(pitch_obj.frequencies[valid_mask], ref_hz=ref_hz)
    tonic_cents = hz_to_cents(np.array([tonic_hz], dtype=float), ref_hz=ref_hz)[0]
    relative_cents[valid_mask] = cents[valid_mask] - tonic_cents
    relative_cents_folded[valid_mask] = ((relative_cents[valid_mask] + 600.0) % 1200.0) - 600.0

    return {
        "track_id": pitch_obj.track_id,
        "tonic_hz": float(tonic_hz),
        "times": times,
        "voiced_mask": voiced_mask,
        "cents": cents,
        "relative_cents": relative_cents,
        "relative_cents_folded": relative_cents_folded,
    }

def generate_basic_tonic_comment(tonic_result: dict, normalized_result: dict) -> str:
    track_id = tonic_result.get("track_id", "unknown track")
    tonic_hz = tonic_result.get("tonic_hz")
    tonic_pc_cents = tonic_result.get("tonic_pc_cents")
    n_voiced_used = tonic_result.get("n_voiced_used", 0)

    relative_cents = normalized_result.get("relative_cents")
    voiced_mask = normalized_result.get("voiced_mask")

    lines = []

    if tonic_hz is None:
        return (
            f"Tonic estimation could not determine a stable tonic for {track_id}. "
            f"The normalized contour is therefore unavailable."
        )

    lines.append(
        f"For {track_id}, the estimated tonic is {tonic_hz:.2f} Hz "
        f"(folded pitch-class center: {tonic_pc_cents:.1f} cents)."
    )
    lines.append(f"This estimate used {n_voiced_used} voiced frames from the contour.")

    if relative_cents is not None and voiced_mask is not None:
        voiced_relative = relative_cents[voiced_mask & np.isfinite(relative_cents)]
        if len(voiced_relative) > 0:
            rel_min = float(np.min(voiced_relative))
            rel_max = float(np.max(voiced_relative))
            rel_median = float(np.median(voiced_relative))
            lines.append(
                f"After tonic normalization, the voiced contour spans roughly "
                f"{rel_min:.1f} to {rel_max:.1f} cents relative to Sa, "
                f"with a median near {rel_median:.1f} cents."
            )

    return " ".join(lines)

def plot_tonic_normalized_contour_snippet(
    normalized_result: dict,
    start_sec: float = 0.0,
    end_sec: float = 20.0,
    folded: bool = False,
    figsize: tuple = (14, 4),
) -> None:
    times = normalized_result["times"]
    voiced_mask = normalized_result["voiced_mask"]

    if folded:
        values = normalized_result["relative_cents_folded"]
        ylabel = "Relative Pitch (folded cents)"
        title_suffix = "Folded"
    else:
        values = normalized_result["relative_cents"]
        ylabel = "Relative Pitch (cents from tonic)"
        title_suffix = "Unfolded"

    snippet_mask = (
        (times >= start_sec)
        & (times <= end_sec)
        & voiced_mask
        & np.isfinite(values)
    )

    snippet_times = times[snippet_mask]
    snippet_values = values[snippet_mask]

    plt.figure(figsize=figsize)

    if len(snippet_times) == 0:
        plt.text(
            0.5,
            0.5,
            "No voiced tonic-normalized samples in this time range.",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.title(f"{title_suffix} tonic-normalized contour snippet")
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()
        return

    plt.plot(snippet_times, snippet_values, linewidth=0.8, color="darkgreen")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    plt.title(
        f"{title_suffix} tonic-normalized contour snippet "
        f"({start_sec:.1f}s to {end_sec:.1f}s)"
    )
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()