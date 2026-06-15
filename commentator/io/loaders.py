"""Input/output helpers for Saraga track loading."""

from __future__ import annotations

from ..core import PitchContour


def initialize_saraga(data_home: str):
    """Initialize the Saraga Hindustani dataset with mirdata."""
    import mirdata
    return mirdata.initialize("saraga_hindustani", data_home=data_home)


def get_track(saraga_dataset, track_id: str):
    """Return the raw mirdata track object for a given track id."""
    return saraga_dataset.track(track_id)


def load_track(track_id: str, saraga_dataset) -> PitchContour:
    """Load a track's pitch contour from an initialized Saraga dataset."""
    track = get_track(saraga_dataset, track_id)
    pitch = getattr(track, "pitch", None)
    if pitch is None:
        raise ValueError(f"Track {track_id!r} does not contain pitch data")

    return PitchContour.from_f0data(
        pitch,
        track_id=track.track_id,
        source="saraga",
    )


def get_duration_for_track(track_id: str, saraga_dataset) -> float:
    """Return track duration in seconds if available."""
    track = get_track(saraga_dataset, track_id)

    duration = getattr(track, "duration", None)
    if duration is not None:
        return float(duration)

    pitch = getattr(track, "pitch", None)
    if pitch is not None and hasattr(pitch, "times") and len(pitch.times) > 0:
        return float(pitch.times[-1])

    raise ValueError(f"Could not determine duration for track {track_id!r}")


def get_pitch_for_track(track_id: str, saraga_dataset) -> PitchContour:
    """Alias for loading a PitchContour for one track id."""
    return load_track(track_id, saraga_dataset)