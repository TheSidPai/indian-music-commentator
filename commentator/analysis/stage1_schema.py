from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np

from .pipeline_one import analyze_pitch_musically


def _safe_float(value):
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def _compute_voiced_range_summary(normalized_result: Dict[str, Any]) -> Dict[str, Any]:
    relative_cents = normalized_result.get("relative_cents")
    voiced_mask = normalized_result.get("voiced_mask")

    if relative_cents is None or voiced_mask is None:
        return {
            "min_cents": None,
            "max_cents": None,
            "median_cents": None,
        }

    valid_mask = voiced_mask & np.isfinite(relative_cents)
    values = relative_cents[valid_mask]

    if len(values) == 0:
        return {
            "min_cents": None,
            "max_cents": None,
            "median_cents": None,
        }

    return {
        "min_cents": float(np.min(values)),
        "max_cents": float(np.max(values)),
        "median_cents": float(np.median(values)),
    }


def _compact_histogram_summary(tonic_result: Dict[str, Any]) -> Dict[str, Any]:
    hist_data = tonic_result.get("hist_data", {})

    return {
        "bin_centers": hist_data.get("bin_centers"),
        "hist": hist_data.get("hist"),
        "hist_smoothed": hist_data.get("hist_smoothed"),
        "bin_size_cents": hist_data.get("bin_size_cents"),
        "ref_hz": hist_data.get("ref_hz"),
    }


def build_stage1_schema(
    pitch_obj,
    raga_label: Optional[str] = None,
    include_artifacts: bool = False,
) -> Dict[str, Any]:
    pipeline_result = analyze_pitch_musically(pitch_obj)

    tonic_result = pipeline_result["tonic_result"]
    normalized_result = pipeline_result["normalized_result"]
    swara_assignment = pipeline_result["swara_assignment"]
    swara_summary = pipeline_result["swara_summary"]

    result = {
        "meta": {
            "track_id": tonic_result.get("track_id"),
            "source": tonic_result.get("source"),
            "input_type": "pitch_contour",
            "raga_label": raga_label,
            "analysis_version": "stage1_v1",
        },
        "tonic": {
            "tonic_hz": _safe_float(tonic_result.get("tonic_hz")),
            "tonic_pc_cents": _safe_float(tonic_result.get("tonic_pc_cents")),
            "n_voiced_used": int(tonic_result.get("n_voiced_used", 0)),
            "target_hz": _safe_float(tonic_result.get("target_hz")),
            "candidate_hz": tonic_result.get("candidate_hz", []),
            "method": tonic_result.get("method"),
        },
        "swara": {
            "n_voiced_frames": int(swara_summary.get("n_voiced_frames", 0)),
            "n_confident_frames": int(swara_summary.get("n_confident_frames", 0)),
            "confident_ratio": _safe_float(swara_summary.get("confident_ratio")),
            "unassigned_frames": int(swara_summary.get("unassigned_frames", 0)),
            "dominant_swaras": swara_summary.get("dominant_swaras", []),
            "least_used_swaras": swara_summary.get("least_used_swaras", []),
            "swara_counts": swara_summary.get("swara_counts", {}),
            "swara_proportions": swara_summary.get("swara_proportions", {}),
        },
        "pitch_summary": {
            "voiced_range_cents": _compute_voiced_range_summary(normalized_result),
            "pitch_histogram": _compact_histogram_summary(tonic_result),

            # Needed for contour / stability / pattern features
            "times": normalized_result.get("times"),
            "voiced_mask": normalized_result.get("voiced_mask"),
            "relative_cents": normalized_result.get("relative_cents"),
            "relative_cents_folded": normalized_result.get("relative_cents_folded"),
        },
        "comments": {
            "basic_comment": pipeline_result.get("comment"),
        },
    }

    if include_artifacts:
        result["artifacts"] = {
            "tonic_histogram": _compact_histogram_summary(tonic_result),
            "normalized_contour": {
                "times": normalized_result.get("times"),
                "voicedmask": normalized_result.get("voicedmask"),
                "relativecents": normalized_result.get("relativecents"),
                "relativecentsfolded": normalized_result.get("relativecentsfolded"),
            },
            "swara_assignment": {
                "assignedswaras": swara_assignment.get("assignedswaras"),
                "nearestswaracents": swara_assignment.get("nearestswaracents"),
                "distancetoswaracents": swara_assignment.get("distancetoswaracents"),
                "confidentmask": swara_assignment.get("confidentmask"),
                "tolerancecents": swara_assignment.get("tolerancecents"),
            },
        }

    return result