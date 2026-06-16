"""
Baseline KNN experiment for raga classification.

Purpose
-------
Use the Stage-1 schema and Stage-2 feature extractor to build a simple,
track-level raga classification baseline.

Pipeline
--------
1. Load a track's PitchContour.
2. Build Stage-1 schema from musical analysis.
3. Convert Stage-1 schema to Stage-2 numeric features.
4. Train and evaluate a KNN classifier.

This module intentionally keeps the baseline simple and interpretable.
"""

from __future__ import annotations

from typing import List, Dict, Any, Callable, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .stage1_schema import build_stage1_schema
from .raga_features import extract_raga_features_from_stage1


TrackConfig = Dict[str, str]
# Expected keys per item:
# {
#   "track_id": "...",
#   "raga_label": "..."
# }


def build_track_feature_dataset(
    tracks: List[TrackConfig],
    get_pitch_fn: Callable[[str], Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Build a track-level feature dataset.

    Parameters
    ----------
    tracks : list of dict
        Each dict must contain:
        - "track_id"
        - "raga_label"
    get_pitch_fn : callable
        get_pitch_fn(track_id) -> PitchContour

    Returns
    -------
    X : np.ndarray, shape (n_tracks, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_tracks,)
        Raga labels.
    track_ids : np.ndarray, shape (n_tracks,)
        Track ids aligned with rows of X.
    feature_names : list[str]
        Names of features in X.
    stage1_records : list[dict]
        The Stage-1 schema objects used to build each track row.
    """
    all_features = []
    all_labels = []
    all_track_ids = []
    stage1_records: List[Dict[str, Any]] = []
    feature_names: List[str] | None = None

    for cfg in tracks:
        track_id = cfg["track_id"]
        raga_label = cfg["raga_label"]

        pitch_obj = get_pitch_fn(track_id)

        stage1 = build_stage1_schema(
            pitch_obj,
            raga_label=raga_label,
            include_artifacts=False,
        )

        feat_vec, feat_names, meta = extract_raga_features_from_stage1(stage1)

        if feature_names is None:
            feature_names = feat_names
        else:
            if feat_names != feature_names:
                raise ValueError(
                    f"Feature name mismatch for track {track_id}. "
                    "All tracks must produce the same feature ordering."
                )

        all_features.append(feat_vec)
        all_labels.append(raga_label)
        all_track_ids.append(track_id)
        stage1_records.append(stage1)

    if len(all_features) == 0:
        X = np.zeros((0, 0), dtype=float)
        y = np.array([], dtype=object)
        track_ids = np.array([], dtype=object)
        return X, y, track_ids, [], []

    X = np.vstack(all_features).astype(float)
    y = np.array(all_labels, dtype=object)
    track_ids = np.array(all_track_ids, dtype=object)

    return X, y, track_ids, feature_names or [], stage1_records

def select_baseline_feature_subset(
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Keep only the musically useful / relatively invariant features
    for the first baseline.
    """

    """
    removed:

        "confident_ratio",

        "min_relative_cents",
        "max_relative_cents",
        "median_relative_cents",
        "range_span_cents",

        "swara_prop_Sa",

        "hist_entropy",
    """

    keep_names = [
        
        "swara_prop_re",
        "swara_prop_Re",
        "swara_prop_ga",
        "swara_prop_Ga",
        "swara_prop_Ma",
        "swara_prop_Ma^",
        "swara_prop_Pa",
        "swara_prop_dha",
        "swara_prop_Dha",
        "swara_prop_ni",
        "swara_prop_Ni",

        "hist_peak_1_cents",
        "hist_peak_1_height",
        "hist_peak_2_cents",
        "hist_peak_2_height",
        "hist_peak_3_cents",
        "hist_peak_3_height",
        
        "hist_concentration",
    ]

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    selected_names = [name for name in keep_names if name in name_to_idx]
    selected_idx = [name_to_idx[name] for name in selected_names]

    X_selected = X[:, selected_idx]
    return X_selected, selected_names

def train_knn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 3,
) -> Pipeline:
    """
    Train a simple standardized KNN classifier.
    """
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_knn_leave_one_track_out(
    tracks: List[TrackConfig],
    get_pitch_fn: Callable[[str], Any],
    n_neighbors: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate the baseline with leave-one-track-out cross-validation.

    For each track:
    - train on all other tracks
    - test on the held-out track
    - record predicted and true label

    Returns
    -------
    result : dict
        {
            "track_results": [...],
            "track_accuracy": float,
            "n_tracks": int,
            "feature_names": list[str],
            "X_shape": tuple,
        }
    """
    X, y, track_ids, feature_names, stage1_records = build_track_feature_dataset(
        tracks=tracks,
        get_pitch_fn=get_pitch_fn,
    )

    X, feature_names = select_baseline_feature_subset(X, feature_names)

    if X.shape[0] == 0:
        return {
            "track_results": [],
            "track_accuracy": 0.0,
            "n_tracks": 0,
            "feature_names": feature_names,
            "X_shape": X.shape,
        }

    unique_tracks = list(track_ids)
    track_results = []

    for i, test_track in enumerate(unique_tracks):
        test_mask = (track_ids == test_track)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            continue

        clf = train_knn_classifier(
            X_train=X_train,
            y_train=y_train,
            n_neighbors=n_neighbors,
        )

        y_pred = clf.predict(X_test)

        predicted_label = y_pred[0]
        true_label = y_test[0]
        correct = bool(predicted_label == true_label)

        track_results.append(
            {
                "track_id": test_track,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": correct,
            }
        )

    n_tracks = len(track_results)
    n_correct = sum(1 for r in track_results if r["correct"])
    track_accuracy = float(n_correct / n_tracks) if n_tracks > 0 else 0.0

    return {
        "track_results": track_results,
        "track_accuracy": track_accuracy,
        "n_tracks": n_tracks,
        "feature_names": feature_names,
        "X_shape": X.shape,
    }


def fit_full_knn_model(
    tracks: List[TrackConfig],
    get_pitch_fn: Callable[[str], Any],
    n_neighbors: int = 3,
) -> Dict[str, Any]:
    """
    Fit a KNN model on the full provided dataset.

    Useful after evaluation when you want a trained baseline model object
    over all currently available tracks.
    """
    X, y, track_ids, feature_names, stage1_records = build_track_feature_dataset(
        tracks=tracks,
        get_pitch_fn=get_pitch_fn,
    )

    if X.shape[0] == 0:
        raise ValueError("No tracks available to train the full KNN model.")

    clf = train_knn_classifier(
        X_train=X,
        y_train=y,
        n_neighbors=n_neighbors,
    )

    return {
        "model": clf,
        "X": X,
        "y": y,
        "track_ids": track_ids,
        "feature_names": feature_names,
        "stage1_records": stage1_records,
    }


def predict_raga_for_track(
    pitch_obj,
    trained_bundle: Dict[str, Any],
    raga_label: str | None = None,
) -> Dict[str, Any]:
    """
    Predict raga for a single new track using a trained full-model bundle.
    """
    stage1 = build_stage1_schema(
        pitch_obj,
        raga_label=raga_label,
        include_artifacts=False,
    )

    feat_vec, feat_names, meta = extract_raga_features_from_stage1(stage1)

    expected_feature_names = trained_bundle["feature_names"]
    if feat_names != expected_feature_names:
        raise ValueError("Feature ordering mismatch between training and inference.")

    clf = trained_bundle["model"]
    pred = clf.predict(feat_vec.reshape(1, -1))[0]

    return {
        "track_id": stage1.get("meta", {}).get("track_id"),
        "predicted_label": pred,
        "stage1": stage1,
        "feature_vector": feat_vec,
    }