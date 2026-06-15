"""
Baseline 3-raga KNN experiment using tonic-normalized pitch features.

Relies on:
- raga_features.extract_segment_features
- An external helper that provides PitchContour for a track_id
  (e.g., via mirdata + your pitch extractor).
"""

from __future__ import annotations
import os

from typing import List, Dict, Any, Tuple, Callable

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from commentator.io import (
    initialize_saraga,
    get_pitch_for_track,
    get_duration_for_track,
)

from .raga_features import extract_segment_features

DATA_HOME = os.path.expanduser("~/mir_projects/data")

saraga = initialize_saraga(DATA_HOME)

TrackConfig = Dict[str, str]
# expected keys: "track_id", "raga_label"

def pitch_fn(track_id):
    return get_pitch_for_track(track_id, saraga)

def duration_fn(track_id):
    return get_duration_for_track(track_id, saraga)

def default_swara_order() -> List[str]:
    """
    Fixed swara order for swara proportion vector.

    Must match get_swara_reference_cents() ordering from swara_analyzer.py.
    """
    return [
        "Sa",
        "re",
        "Re",
        "ga",
        "Ga",
        "Ma",
        "Ma^",
        "Pa",
        "dha",
        "Dha",
        "ni",
        "Ni",
    ]


def build_segments_for_track(
    total_duration: float,
    segment_length: float = 60.0,
    hop: float = 60.0,
) -> List[Tuple[float, float]]:
    """
    Simple uniform segmentation of [0, total_duration).

    You can replace this with hand-picked segments later if needed.
    """
    segments = []
    start = 0.0
    while start + segment_length <= total_duration:
        end = start + segment_length
        segments.append((start, end))
        start += hop
    return segments


def build_feature_dataset(
    tracks: List[TrackConfig],
    get_pitch_fn: Callable[[str], Any],
    get_duration_fn: Callable[[str], float],
    segment_length: float = 60.0,
    hop: float = 60.0,
    ref_hz: float = 55.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix X, label vector y, and track_ids per segment.

    Parameters
    ----------
    tracks : list of dict
        [{"track_id": ..., "raga_label": ...}, ...]
    get_pitch_fn : callable
        get_pitch_fn(track_id) -> PitchContour
    get_duration_fn : callable
        get_duration_fn(track_id) -> float (seconds)
    segment_length : float
        Length of each segment in seconds.
    hop : float
        Hop between consecutive segment start times.

    Returns
    -------
    X : np.ndarray, shape (n_segments, n_features)
    y : np.ndarray, shape (n_segments,)
        String labels (raga names).
    segment_track_ids : np.ndarray, shape (n_segments,)
        Track_id for each segment (for track-level evaluation).
    """
    all_features = []
    all_labels = []
    all_track_ids = []

    swara_order = default_swara_order()

    for cfg in tracks:
        track_id = cfg["track_id"]
        raga_label = cfg["raga_label"]

        # 1. Obtain pitch + duration
        pitch_obj = get_pitch_fn(track_id)
        duration = get_duration_fn(track_id)

        # 2. Decide segments for this track
        segments = build_segments_for_track(
            total_duration=duration,
            segment_length=segment_length,
            hop=hop,
        )

        # 3. (Optional) track-level tonic estimation to reuse across segments
        #    For v1, we can reuse per-segment tonic.
        #    Later, you can estimate once on full pitch_obj and pass as reuse_tonic_result.

        for (start, end) in segments:
            feat, meta = extract_segment_features(
                pitch_obj,
                start_time=start,
                end_time=end,
                ref_hz=ref_hz,
                swara_order=swara_order,
            )

            all_features.append(feat)
            all_labels.append(raga_label)
            all_track_ids.append(track_id)

    X = np.vstack(all_features) if all_features else np.zeros((0, 1), dtype=float)
    y = np.array(all_labels, dtype=object)
    segment_track_ids = np.array(all_track_ids, dtype=object)

    return X, y, segment_track_ids


def run_knn_leave_one_track_out(
    tracks: List[TrackConfig],
    get_pitch_fn: Callable[[str], Any],
    get_duration_fn: Callable[[str], float],
    segment_length: float = 60.0,
    hop: float = 60.0,
    n_neighbors: int = 3,
) -> Dict[str, Any]:
    """
    Run a KNN baseline with leave-one-track-out evaluation.

    For each track:
        - Train on all segments from other tracks.
        - Test on all segments from this track.
        - Aggregate segment predictions by majority vote -> track-level raga.
    """
    # Build full dataset once
    X, y, segment_track_ids = build_feature_dataset(
        tracks,
        get_pitch_fn=get_pitch_fn,
        get_duration_fn=get_duration_fn,
        segment_length=segment_length,
        hop=hop,
    )

    unique_tracks = [cfg["track_id"] for cfg in tracks]
    track_to_raga = {cfg["track_id"]: cfg["raga_label"] for cfg in tracks}

    track_results = []

    for test_track in unique_tracks:
        # Boolean masks for train/test segments
        test_mask = (segment_track_ids == test_track)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if X_test.shape[0] == 0 or X_train.shape[0] == 0:
            continue

        # KNN pipeline
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ]
        )
        clf.fit(X_train, y_train)

        # Segment-level predictions
        y_pred_segments = clf.predict(X_test)

        # Majority vote for this track
        unique, counts = np.unique(y_pred_segments, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        true_label = track_to_raga[test_track]
        correct = (majority_label == true_label)

        track_results.append(
            {
                "track_id": test_track,
                "true_label": true_label,
                "predicted_label": majority_label,
                "segment_predictions": y_pred_segments,
                "segment_true_labels": y_test,
            }
        )

    # Compute track-level accuracy
    n_tracks = len(track_results)
    n_correct = sum(1 for r in track_results if r["true_label"] == r["predicted_label"])
    track_accuracy = float(n_correct / n_tracks) if n_tracks > 0 else 0.0

    return {
        "track_results": track_results,
        "track_accuracy": track_accuracy,
        "n_tracks": n_tracks,
    }