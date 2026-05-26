"""Pitch contour data structures and helpers.

The implementation is adapted from the working notebook prototype so it can be
used from regular Python modules without changing the notebook itself.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PitchContour:
    """Container for sampled pitch tracking data."""

    times: np.ndarray
    frequencies: np.ndarray
    voiced_mask: np.ndarray
    confidence: Optional[np.ndarray] = None
    source: str = "unknown"
    track_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Coerce inputs to arrays and validate aligned lengths."""

        self.times = np.asarray(self.times, dtype=float)
        self.frequencies = np.asarray(self.frequencies, dtype=float)
        self.voiced_mask = np.asarray(self.voiced_mask, dtype=bool)

        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=float)

        n_frames = len(self.times)
        if len(self.frequencies) != n_frames:
            raise ValueError("times and frequencies must have the same length")
        if len(self.voiced_mask) != n_frames:
            raise ValueError("times and voiced_mask must have the same length")
        if self.confidence is not None and len(self.confidence) != n_frames:
            raise ValueError("times and confidence must have the same length")

    @property
    def duration(self) -> float:
        """Return the final timestamp or 0.0 for empty contours."""

        return float(self.times[-1]) if len(self.times) > 0 else 0.0

    @property
    def voiced_times(self) -> np.ndarray:
        """Return timestamps for voiced frames."""

        return self.times[self.voiced_mask]

    @property
    def voiced_frequencies(self) -> np.ndarray:
        """Return frequencies for voiced frames."""

        return self.frequencies[self.voiced_mask]

    @classmethod
    def from_f0data(
        cls,
        f0data,
        track_id: Optional[str] = None,
        source: str = "saraga",
    ) -> "PitchContour":
        """Build a contour from a mirdata F0 annotation object."""

        times = np.asarray(f0data.times, dtype=float)
        freqs = np.asarray(f0data.frequencies, dtype=float)

        if hasattr(f0data, "voicing") and f0data.voicing is not None:
            voiced_mask = np.asarray(f0data.voicing).astype(bool)
        else:
            voiced_mask = freqs > 0

        confidence = None
        if hasattr(f0data, "confidence") and f0data.confidence is not None:
            confidence = np.asarray(f0data.confidence, dtype=float)

        return cls(
            times=times,
            frequencies=freqs,
            voiced_mask=voiced_mask,
            confidence=confidence,
            source=source,
            track_id=track_id,
        )

    def summary(self) -> dict:
        """Return a compact statistical summary of the contour."""

        voiced_freqs = self.voiced_frequencies

        if len(self.times) == 0:
            return {
                "source": self.source,
                "track_id": self.track_id,
                "n_frames": 0,
                "duration_sec": 0.0,
                "n_voiced": 0,
                "voiced_ratio": 0.0,
                "min_hz": None,
                "max_hz": None,
                "mean_hz": None,
                "median_hz": None,
            }

        return {
            "source": self.source,
            "track_id": self.track_id,
            "n_frames": len(self.times),
            "duration_sec": float(self.duration),
            "n_voiced": int(self.voiced_mask.sum()),
            "voiced_ratio": float(self.voiced_mask.mean()),
            "min_hz": float(voiced_freqs.min()) if len(voiced_freqs) > 0 else None,
            "max_hz": float(voiced_freqs.max()) if len(voiced_freqs) > 0 else None,
            "mean_hz": float(voiced_freqs.mean()) if len(voiced_freqs) > 0 else None,
            "median_hz": float(np.median(voiced_freqs)) if len(voiced_freqs) > 0 else None,
        }
