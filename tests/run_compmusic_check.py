"""Smoke test for the CompMusic Indian Art Music Raga Recognition Dataset
(mirdata id: "compmusic_raga"), Hindustani subset (HMD).

Purpose
-------
Check whether this larger dataset (300 recordings, 30 ragas, 10 recordings
per raga) can scale up the current saraga_hindustani pipeline (108 tracks,
only 6 ragas with >=2 usable tracks) without any changes to the Stage-1/
Stage-2 feature-extraction code.

The dataset-specific loading lives in commentator/io/compmusic.py (including
the reason mirdata's own loader cannot be used for the Hindustani half);
this script only exercises it.

Usage
-----
    .venv/bin/python tests/run_compmusic_check.py
    .venv/bin/python tests/run_compmusic_check.py --download   # first run, ~3.4 GB
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commentator.io.compmusic import CompMusicHindustani
from commentator.analysis.stage1_schema import build_stage1_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="fetch the ~3.4 GB features archive if absent")
    parser.add_argument("--data-home", default=None)
    args = parser.parse_args()

    dataset = CompMusicHindustani(data_home=args.data_home, download=args.download)

    records = dataset.list_tracks()
    print(f"Hindustani tracks: {len(records)}")

    raga_counts = Counter(r["raga_label"] for r in records)
    print(f"Ragas: {len(raga_counts)}")
    for raga, count in sorted(raga_counts.items()):
        print(f"  {raga}: {count}")

    if not records:
        print("No Hindustani tracks found -- stopping.")
        return

    # Sanity check: does this feed cleanly into the existing Stage-1 pipeline?
    record = records[0]
    track_id = record["track_id"]
    pitch_obj = dataset.get_pitch(track_id)
    annotated_tonic = dataset.get_tonic(track_id)

    print()
    print(f"Sample track: {track_id}")
    print(f"  raga:   {record['raga_label']}")
    print(f"  artist: {record['artist']}")
    print(f"  contour: {pitch_obj.summary()}")

    stage1 = build_stage1_schema(
        pitch_obj,
        raga_label=record["raga_label"],
        include_artifacts=False,
    )
    estimated_tonic = stage1["tonic"]["tonic_hz"]

    print()
    print("Stage-1 estimated tonic_hz:", estimated_tonic)
    print("Dataset annotated tonic_hz:", annotated_tonic)
    if annotated_tonic and estimated_tonic:
        cents = 1200 * np.log2(estimated_tonic / annotated_tonic)
        print(f"Difference: {cents:+.1f} cents")
    print("Stage-1 dominant swaras:", stage1["swara"]["dominant_swaras"])
    print("Stage-1 comment:", stage1["comments"]["basic_comment"])


if __name__ == "__main__":
    main()
