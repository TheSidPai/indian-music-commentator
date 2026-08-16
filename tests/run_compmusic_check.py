"""Smoke test for the CompMusic Indian Art Music Raga Recognition Dataset
(mirdata id: "compmusic_raga"), Hindustani subset (HMD).

Purpose
-------
Check whether this larger dataset (300 recordings, 30 ragas, 10 recordings
per raga) can scale up the current saraga_hindustani pipeline (108 tracks,
only 6 ragas with >=2 usable tracks) without any changes to the Stage-1/
Stage-2 feature-extraction code.

What this does
---------------
1. Initializes the dataset and downloads the features/metadata archive
   (~3.6 GB, freely available on Zenodo, no login/API key required -- this
   is the "features" DOI, not the restricted-access "audio" one).
2. Lists the Hindustani-tradition track ids and their raga distribution,
   as a sanity check against the documented "10 recordings per raga".
3. Loads one sample track's pitch contour and wraps it in the existing
   PitchContour dataclass (commentator/core/pitch_contour.py), the same
   way commentator/io/loaders.py does for saraga_hindustani.
4. Runs that PitchContour through the existing build_stage1_schema(...)
   entry point, to confirm the Stage-1/Stage-2 pipeline needs zero changes
   to consume this dataset.

Note: raw audio is intentionally not touched here. This dataset's audio
files are ~9.2 TB combined (both traditions) and access-restricted; the
existing classification pipeline only ever consumes pitch/tonic arrays,
so there's no need to request audio access for this part of the project.
"""

from __future__ import annotations

import os
from collections import Counter

import mirdata

from commentator.core.pitch_contour import PitchContour
from commentator.analysis.stage1_schema import build_stage1_schema

DATA_HOME = os.environ.get(
    "COMPMUSIC_RAGA_DATA_HOME",
    os.path.expanduser("~/mir_projects/data/compmusic_raga"),
)


def load_dataset():
    """Initialize compmusic_raga and download the free features archive.

    dataset.download() only fetches the features/metadata zip (mirdata's
    REMOTES config excludes audio entirely, since that requires a manual
    access request). Safe to call repeatedly -- mirdata skips re-downloading
    once the checksum matches.
    """
    dataset = mirdata.initialize("compmusic_raga", data_home=DATA_HOME)
    dataset.download()
    dataset.validate()
    return dataset


def list_hindustani_tracks(dataset) -> list[str]:
    """Return track_ids whose tradition is 'hindustani' (vs 'carnatic')."""
    return [
        track_id
        for track_id in dataset.track_ids
        if dataset.track(track_id).tradition == "hindustani"
    ]


def summarize_raga_distribution(dataset, track_ids: list[str]) -> Counter:
    """Count tracks per raga label, to check the '10 per raga' claim."""
    counts: Counter = Counter()
    for track_id in track_ids:
        counts[dataset.track(track_id).raga] += 1
    return counts


def load_pitch_contour(dataset, track_id: str) -> PitchContour:
    """Wrap a compmusic_raga track's pitch into the existing PitchContour.

    compmusic_raga's `track.pitch` is an F0Data object with the same
    times/frequencies/voicing shape as saraga_hindustani's, so
    PitchContour.from_f0data(...) works unchanged -- mirrors
    commentator/io/loaders.py's load_track(...) for Saraga.
    """
    track = dataset.track(track_id)
    return PitchContour.from_f0data(
        track.pitch,
        track_id=track.track_id,
        source="compmusic_raga",
    )


def main() -> None:
    dataset = load_dataset()

    hindustani_ids = list_hindustani_tracks(dataset)
    print(f"Hindustani tracks: {len(hindustani_ids)}")

    raga_counts = summarize_raga_distribution(dataset, hindustani_ids)
    print(f"Ragas: {len(raga_counts)}")
    for raga, count in sorted(raga_counts.items()):
        print(f"  {raga}: {count}")

    if not hindustani_ids:
        print("No Hindustani tracks found -- stopping.")
        return

    sample_track_id = hindustani_ids[0]
    track = dataset.track(sample_track_id)
    pitch_obj = load_pitch_contour(dataset, sample_track_id)

    print()
    print(f"Sample track: {sample_track_id}  (raga={track.raga})")
    print(pitch_obj.summary())

    # Sanity check: does this feed cleanly into the existing Stage-1 pipeline?
    stage1 = build_stage1_schema(
        pitch_obj,
        raga_label=track.raga,
        include_artifacts=False,
    )
    print()
    print("Stage-1 tonic_hz:", stage1["tonic"]["tonic_hz"])
    print("Stage-1 dominant swaras:", stage1["swara"]["dominant_swaras"])
    print("Stage-1 comment:", stage1["comments"]["basic_comment"])


if __name__ == "__main__":
    main()
