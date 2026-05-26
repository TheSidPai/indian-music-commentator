"""Input/output helpers for Saraga track loading."""

from __future__ import annotations

from ..core import PitchContour


def initialize_saraga(data_home: str):
    """Initialize the Saraga Hindustani dataset with mirdata."""

    import mirdata

    return mirdata.initialize("saraga_hindustani", data_home=data_home)


def load_track(track_id: str, saraga_dataset) -> PitchContour:
    """Load a track's pitch contour from an initialized Saraga dataset."""

    track = saraga_dataset.track(track_id)
    pitch = getattr(track, "pitch", None)
    if pitch is None:
        raise ValueError(f"Track {track_id!r} does not contain pitch data")

    return PitchContour.from_f0data(pitch, track_id=track.track_id, source="saraga")
