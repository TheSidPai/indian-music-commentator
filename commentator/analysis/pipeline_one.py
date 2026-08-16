"""
End-to-end analysis pipeline for tonic and basic swara usage.

Takes a PitchContour, estimates tonic, normalizes to tonic,
assigns swaras, summarizes usage, and generates a short comment.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

from .tonic_estimator import estimate_tonic, normalize_to_tonic, hz_to_cents
from .swara_analyzer import (
    assign_swaras_to_contour,
    summarize_swara_distribution,
    generate_basic_swara_comment,
)


def analyze_pitch_musically(
    pitch_obj,
    *,
    ref_hz: float = 55.0,
    bin_size_cents: float = 10.0,
    min_hz: float = 60.0,
    max_hz: float = 1000.0,
    smooth_kernel_size: int = 7,
    candidate_min_hz: float = 80.0,
    candidate_max_hz: float = 400.0,
    swara_tolerance_cents: float = 35.0,
    tonic_hz: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run the full tonic + swara analysis pipeline on a PitchContour.

    Parameters
    ----------
    pitch_obj
        PitchContour-like object with at least:
        - times
        - frequencies
        - voiced_mask
        - voiced_frequencies
        - source
        - track_id
    ref_hz
        Reference frequency for cents conversion in tonic estimation.
    bin_size_cents
        Histogram bin size for folded pitch-class histogram.
    min_hz, max_hz
        Frequency bounds for selecting voiced frequencies.
    smooth_kernel_size
        Smoothing kernel size for pitch-class histogram.
    candidate_min_hz, candidate_max_hz
        Allowed tonic octave range in Hz.
    swara_tolerance_cents
        Tolerance for assigning contour points to nearest swara.
    tonic_hz
        Optional known tonic in Hz (e.g. a dataset's annotated tonic). When
        given, it replaces the estimated tonic for normalization and swara
        assignment; the histogram is still computed, so histogram-derived
        features are unaffected, and the value that *would* have been
        estimated is preserved under `tonic_result["estimated_tonic_hz"]`
        for comparison. See tests/run_tonic_validation.py for why this
        matters: on the 6-raga Saraga set, 50.2% of 30s segments estimate a
        pitch-class-wrong tonic.

    Returns
    -------
    result : dict
        {
            "pitch_source": str,
            "track_id": str,
            "tonic_result": dict,
            "normalized_result": dict,
            "swara_assignment": dict,
            "swara_summary": dict,
            "comment": str,
        }
    """

    # 1. Tonic estimation
    tonic_result = estimate_tonic(
        pitch_obj,
        ref_hz=ref_hz,
        bin_size_cents=bin_size_cents,
        min_hz=min_hz,
        max_hz=max_hz,
        smooth_kernel_size=smooth_kernel_size,
        candidate_min_hz=candidate_min_hz,
        candidate_max_hz=candidate_max_hz,
    )

    # 1b. Optionally override the estimate with a known tonic. The histogram
    # in tonic_result is left untouched (it is tonic-independent), so only
    # normalization and swara assignment change.
    if tonic_hz is not None:
        tonic_result = {
            **tonic_result,
            "tonic_hz": float(tonic_hz),
            "tonic_pc_cents": float(
                hz_to_cents(np.array([tonic_hz], dtype=float), ref_hz=ref_hz)[0] % 1200.0
            ),
            "estimated_tonic_hz": tonic_result.get("tonic_hz"),
            "method": "annotated_tonic",
        }

    # 2. Tonic-normalized contour
    normalized_result = normalize_to_tonic(
        pitch_obj,
        tonic_result=tonic_result,
        ref_hz=ref_hz,
    )

    # 3. Swara assignment on folded, tonic-normalized cents
    swara_assignment = assign_swaras_to_contour(
        normalized_result,
        tolerance_cents=swara_tolerance_cents,
    )

    # 4. Swara distribution summary
    swara_summary = summarize_swara_distribution(swara_assignment)

    # 5. Textual comment
    comment = generate_basic_swara_comment(tonic_result, swara_summary)

    return {
        "pitch_source": getattr(pitch_obj, "source", None),
        "track_id": getattr(pitch_obj, "track_id", None),
        "tonic_result": tonic_result,
        "normalized_result": normalized_result,
        "swara_assignment": swara_assignment,
        "swara_summary": swara_summary,
        "comment": comment,
    }