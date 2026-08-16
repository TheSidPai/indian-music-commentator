"""I/O helpers.

Each dataset gets one adapter module exposing the same interface:

    name: str
    list_tracks() -> list[dict]          # [{"track_id", "raga_label", ...}]
    get_pitch(track_id) -> PitchContour
    get_tonic(track_id) -> float | None  # annotated tonic, not estimated

so pipeline code (e.g. build_segment_feature_dataset) can be pointed at
either dataset without changes.
"""

from .saraga import (
    SaragaHindustani,
    initialize_saraga,
    get_track,
    load_track,
    get_pitch_for_track,
    get_duration_for_track,
    get_tonic_for_track,
)
from .compmusic import CompMusicHindustani

__all__ = [
    # dataset adapters
    "SaragaHindustani",
    "CompMusicHindustani",
    # Saraga module-level helpers (used by the existing driver scripts)
    "initialize_saraga",
    "get_track",
    "load_track",
    "get_pitch_for_track",
    "get_duration_for_track",
    "get_tonic_for_track",
]
