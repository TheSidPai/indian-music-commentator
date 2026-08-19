"""Validate Stage-1 tonic estimation against Saraga's annotated tonic.

Why
---
The Stage-1 pipeline estimates the tonic from the pitch histogram. Saraga
ships a ground-truth tonic for every track (`track.tonic`, from the
`ctonic_path` annotation), which the pipeline has never used -- so the
estimator's real error rate has never been measured. The HMD scale-up
surfaced apparent tonic errors, prompting this check on the existing data.

What it measures
----------------
Two levels, because they answer different questions:

1. Track level -- estimate the tonic from the full contour and compare to
   the annotation. This characterizes the estimator in the best case (most
   data available).

2. Segment level -- slice each track into the same 30s/20s-hop windows the
   classifier is trained on and estimate a tonic per segment. This is what
   actually matters: `build_segment_feature_dataset` calls
   `build_stage1_schema` per segment, so every segment carries its own,
   independently estimated tonic. A track-level check would understate the
   error the classifier is exposed to.

Interpreting the two error columns
----------------------------------
Swara assignment folds cents mod 1200 (swara_analyzer.py), so a pure octave
error leaves `swara_prop_*` untouched while still corrupting `tonic_hz` and
the relative-cents range features. A pitch-class error (e.g. mistaking Pa
for Sa) shifts every swara assignment and invalidates the whole vector.
Hence both raw and octave-folded accuracy are reported.

Usage
-----
    .venv/bin/python scripts/run_tonic_validation.py            # 13 experiment tracks
    .venv/bin/python scripts/run_tonic_validation.py --all      # every Saraga track
    .venv/bin/python scripts/run_tonic_validation.py --skip-segments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commentator.io.saraga import initialize_saraga, get_pitch_for_track, get_track
from commentator.analysis.stage1_schema import build_stage1_schema
from commentator.analysis.segment_dataset import slice_pitch_contour

DATA_HOME = "/home/thesidpai/mir_projects/data"

# The tracks the reported 0.9051 RF accuracy was measured on.
EXPERIMENT_TRACKS = [
    "27_Raag_Bihag", "81_Raag_Bihag",
    "8_Raag_Kedar", "84_Raag_Kedar",
    "83_Raag_Bhoopali", "105_Raag_Bhoopali",
    "20_Raag_Abhogi", "44_Raag_Abhogi",
    "0_Raag_Shree", "37_Raag_Shree",
    "10_Raag_Lalit", "33_Raag_Lalit", "104_Raga_Lalit_-_Khayal",
]

TOLERANCE_CENTS = 50.0


def cents_error(estimated_hz: float, annotated_hz: float) -> float:
    """Signed cents difference between an estimate and the annotation."""
    return 1200.0 * float(np.log2(estimated_hz / annotated_hz))


def fold_to_octave(cents: float) -> float:
    """Fold a cents error into (-600, +600], collapsing octave errors to ~0."""
    folded = cents % 1200.0
    return folded - 1200.0 if folded > 600.0 else folded


def estimate_tonic_hz(pitch_obj, raga_label: str | None = None) -> float | None:
    """Run Stage-1 on a contour and return just the estimated tonic."""
    stage1 = build_stage1_schema(
        pitch_obj, raga_label=raga_label, include_artifacts=False
    )
    return stage1["tonic"]["tonic_hz"]


def segment_bounds(duration_s: float, segment_length_s: float, hop_s: float,
                   min_duration_s: float) -> list[tuple[float, float]]:
    """Reproduce the windowing build_segment_feature_dataset uses."""
    bounds = []
    start = 0.0
    while start < duration_s:
        end = min(start + segment_length_s, duration_s)
        if end - start >= min_duration_s:
            bounds.append((start, end))
        if end >= duration_s:
            break
        start += hop_s
    return bounds


def summarize(errors: list[float], label: str) -> None:
    """Print raw vs octave-folded accuracy for a list of cents errors."""
    if not errors:
        print(f"{label}: no measurements")
        return

    arr = np.array(errors)
    folded = np.array([fold_to_octave(e) for e in arr])

    raw_ok = int((np.abs(arr) < TOLERANCE_CENTS).sum())
    folded_ok = int((np.abs(folded) < TOLERANCE_CENTS).sum())
    n = len(arr)

    print(f"\n--- {label} (n={n}) ---")
    print(f"  within +/-{TOLERANCE_CENTS:.0f} cents (raw):           "
          f"{raw_ok}/{n} ({100 * raw_ok / n:.1f}%)")
    print(f"  within +/-{TOLERANCE_CENTS:.0f} cents (octave-folded): "
          f"{folded_ok}/{n} ({100 * folded_ok / n:.1f}%)")
    print(f"  octave-only errors:  {folded_ok - raw_ok}")
    print(f"  pitch-class errors:  {n - folded_ok} "
          f"({100 * (n - folded_ok) / n:.1f}% -- these corrupt swara assignment)")
    print(f"  median |error| (folded): {np.median(np.abs(folded)):.1f} cents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="check every Saraga track, not just the 13 experiment tracks")
    parser.add_argument("--skip-segments", action="store_true",
                        help="track-level check only (much faster)")
    parser.add_argument("--segment-length", type=float, default=30.0)
    parser.add_argument("--hop", type=float, default=20.0)
    parser.add_argument("--min-duration", type=float, default=15.0)
    args = parser.parse_args()

    saraga = initialize_saraga(DATA_HOME)
    track_ids = list(saraga.track_ids) if args.all else EXPERIMENT_TRACKS

    print(f"Validating tonic estimation on {len(track_ids)} track(s)")
    print(f"{'track':<32} {'annot':>8} {'est':>8} {'cents':>8} {'folded':>8}  verdict")
    print("-" * 82)

    track_errors: list[float] = []
    segment_errors: list[float] = []
    per_track_segment_errors: dict[str, list[float]] = {}

    for track_id in track_ids:
        try:
            annotated = getattr(get_track(saraga, track_id), "tonic", None)
            if not annotated:
                print(f"{track_id:<32} {'--':>8}  no annotated tonic, skipped")
                continue

            pitch_obj = get_pitch_for_track(track_id, saraga)
            estimated = estimate_tonic_hz(pitch_obj)
            if not estimated:
                print(f"{track_id:<32} {annotated:>8.2f}  estimation returned nothing")
                continue

            err = cents_error(estimated, annotated)
            folded = fold_to_octave(err)
            track_errors.append(err)

            if abs(err) < TOLERANCE_CENTS:
                verdict = "ok"
            elif abs(folded) < TOLERANCE_CENTS:
                verdict = "OCTAVE"
            else:
                verdict = "PITCH-CLASS"

            print(f"{track_id:<32} {annotated:>8.2f} {estimated:>8.2f} "
                  f"{err:>+8.1f} {folded:>+8.1f}  {verdict}")

            if args.skip_segments:
                continue

            # Segment level: what the classifier actually sees.
            errs = []
            for start, end in segment_bounds(
                pitch_obj.duration, args.segment_length, args.hop, args.min_duration
            ):
                try:
                    seg = slice_pitch_contour(pitch_obj, start, end)
                    seg_est = estimate_tonic_hz(seg)
                    if seg_est:
                        errs.append(cents_error(seg_est, annotated))
                except Exception:
                    continue

            segment_errors.extend(errs)
            per_track_segment_errors[track_id] = errs

            if errs:
                folded_errs = np.array([fold_to_octave(e) for e in errs])
                ok = int((np.abs(folded_errs) < TOLERANCE_CENTS).sum())
                print(f"{'':<32} segments: {ok}/{len(errs)} "
                      f"pitch-class-correct ({100 * ok / len(errs):.0f}%)")

        except Exception as e:
            print(f"{track_id:<32} ERROR {type(e).__name__}: {e}")

    print("\n" + "=" * 82)
    summarize(track_errors, "TRACK LEVEL (full contour)")
    if not args.skip_segments:
        summarize(segment_errors, "SEGMENT LEVEL (what the classifier trains on)")

        worst = sorted(
            (
                (tid, 100 * sum(abs(fold_to_octave(e)) >= TOLERANCE_CENTS for e in errs) / len(errs))
                for tid, errs in per_track_segment_errors.items() if errs
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        print("\n  worst tracks by % of segments with pitch-class errors:")
        for tid, pct in worst[:5]:
            print(f"    {tid:<32} {pct:5.1f}%")


if __name__ == "__main__":
    main()
