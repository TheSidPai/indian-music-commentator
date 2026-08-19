"""Input/output helpers for Saraga Hindustani track loading.

Exposes both the original module-level functions (used by the existing
driver scripts) and a `SaragaHindustani` adapter presenting the same
list_tracks/get_pitch/get_tonic interface as `io.compmusic`, so pipeline
code can be pointed at either dataset without changes.
"""

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


def get_tonic_for_track(track_id: str, saraga_dataset) -> float | None:
    """Return Saraga's annotated tonic in Hz, or None if absent.

    Saraga ships a ground-truth tonic per track (the `ctonic` annotation);
    it is not the value Stage-1 estimates. See scripts/run_tonic_validation.py
    for the comparison between the two.
    """
    tonic = getattr(get_track(saraga_dataset, track_id), "tonic", None)
    return float(tonic) if tonic else None


class SaragaHindustani:
    """Adapter over the Saraga Hindustani dataset.

    Mirrors `io.compmusic.CompMusicHindustani`, so anything written against
    one works against the other.
    """

    name = "saraga"

    def __init__(self, data_home: str):
        self.data_home = data_home
        self._dataset = initialize_saraga(data_home)

    @property
    def dataset(self):
        """The underlying mirdata dataset, for dataset-specific needs."""
        return self._dataset

    def list_tracks(self) -> list[dict]:
        """Return [{"track_id", "raga_label"}] for every track.

        The raga label follows the repo's `Raag_<Name>` convention, derived
        from the track id rather than mirdata's metadata title.
        """
        records = []
        for track_id in self._dataset.track_ids:
            parts = track_id.split("_", 1)
            raga_label = parts[1] if len(parts) == 2 else track_id
            records.append({"track_id": track_id, "raga_label": raga_label})
        return records

    def get_pitch(self, track_id: str) -> PitchContour:
        return load_track(track_id, self._dataset)

    def get_tonic(self, track_id: str) -> float | None:
        return get_tonic_for_track(track_id, self._dataset)