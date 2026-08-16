from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from commentator.io.saraga import (
    initialize_saraga,
    get_pitch_for_track,
    get_tonic_for_track,
)
from commentator.analysis.segment_dataset import build_segment_feature_dataset

DATA_HOME = "/home/thesidpai/mir_projects/data"  # change if needed

TRACKS = [
    # Bihag
    {"track_id": "27_Raag_Bihag", "raga_label": "Raag_Bihag"},
    {"track_id": "81_Raag_Bihag", "raga_label": "Raag_Bihag"},

    # Kedar
    {"track_id": "8_Raag_Kedar", "raga_label": "Raag_Kedar"},
    {"track_id": "84_Raag_Kedar", "raga_label": "Raag_Kedar"},

    # Bhoopali
    {"track_id": "83_Raag_Bhoopali", "raga_label": "Raag_Bhoopali"},
    {"track_id": "105_Raag_Bhoopali", "raga_label": "Raag_Bhoopali"},

    # Abhogi
    {"track_id": "20_Raag_Abhogi", "raga_label": "Raag_Abhogi"},
    {"track_id": "44_Raag_Abhogi", "raga_label": "Raag_Abhogi"},

    # Shree
    {"track_id": "0_Raag_Shree", "raga_label": "Raag_Shree"},
    {"track_id": "37_Raag_Shree", "raga_label": "Raag_Shree"},

    # Lalit
    {"track_id": "10_Raag_Lalit", "raga_label": "Raag_Lalit"},
    {"track_id": "33_Raag_Lalit", "raga_label": "Raag_Lalit"},
    {"track_id": "104_Raga_Lalit_-_Khayal", "raga_label": "Raag_Lalit"},
]


def majority_vote(values: list[str]) -> str:
    if not values:
        return "UNKNOWN"
    return Counter(values).most_common(1)[0][0]


def print_segment_confusion(y_true, y_pred):
    labels = sorted(set(str(x) for x in y_true) | set(str(x) for x in y_pred))
    counts = {t: defaultdict(int) for t in labels}

    for t, p in zip(y_true, y_pred):
        counts[str(t)][str(p)] += 1

    print("\n=== Segment-level confusion matrix ===")
    header = "true\\pred".ljust(18) + "".join(lbl[:16].ljust(18) for lbl in labels)
    print(header)

    for t in labels:
        row = t[:16].ljust(18)
        for p in labels:
            row += str(counts[t][p]).ljust(18)
        print(row)

    print("\n=== Segment-level confusion proportions ===")
    print(header)
    for t in labels:
        total = sum(counts[t].values())
        row = t[:16].ljust(18)
        for p in labels:
            val = counts[t][p] / total if total > 0 else 0.0
            row += f"{val:.2f}".ljust(18)
        print(row)


def evaluate_model(model_name, clf, X, y, groups, valid_records):
    logo = LeaveOneGroupOut()

    track_results = []
    segment_correct = []
    all_y_true = []
    all_y_pred = []

    print(f"\n\n===== {model_name} =====")
    print("=== Leave-one-track-out results ===")

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = [y[i] for i in train_idx]
        y_test = [str(y[i]) for i in test_idx]

        test_records = [valid_records[i] for i in test_idx]
        held_out_track = test_records[0]["track_id"]
        true_label = str(test_records[0]["raga_label"])

        clf.fit(X_train, y_train)
        seg_preds = [str(p) for p in clf.predict(X_test)]

        all_y_true.extend(y_test)
        all_y_pred.extend(seg_preds)

        for pred, truth in zip(seg_preds, y_test):
            segment_correct.append(pred == truth)

        vote_counts = {str(k): v for k, v in Counter(seg_preds).items()}
        track_pred = majority_vote(seg_preds)
        track_correct = (track_pred == true_label)
        seg_acc = sum(p == t for p, t in zip(seg_preds, y_test)) / len(y_test)

        result = {
            "track_id": held_out_track,
            "true_label": true_label,
            "track_pred": track_pred,
            "track_correct": track_correct,
            "n_segments": len(test_idx),
            "segment_accuracy": seg_acc,
            "vote_counts": vote_counts,
        }
        track_results.append(result)

        print(f"\nHeld-out track: {held_out_track}")
        print(f" True label: {true_label}")
        print(f" Pred label: {track_pred}")
        print(f" Track correct: {track_correct}")
        print(f" Num segments: {len(test_idx)}")
        print(f" Segment acc: {seg_acc:.3f}")
        print(f" Vote counts: {vote_counts}")

    track_acc = sum(r["track_correct"] for r in track_results) / len(track_results)
    seg_acc = sum(segment_correct) / len(segment_correct)

    print("\n=== Summary ===")
    print(f"Track-level accuracy: {track_acc:.3f}")
    print(f"Segment-level accuracy: {seg_acc:.3f}")

    print("\n=== Full track results ===")
    for r in track_results:
        print(r)

    print_segment_confusion(all_y_true, all_y_pred)

    print("\n=== Classification report ===")
    print(classification_report(all_y_true, all_y_pred, digits=4))

    return {
        "track_results": track_results,
        "track_accuracy": track_acc,
        "segment_accuracy": seg_acc,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotated-tonic",
        action="store_true",
        help="use Saraga's annotated tonic instead of estimating one per segment "
             "(see tests/run_tonic_validation.py for why this matters)",
    )
    args = parser.parse_args()

    saraga = initialize_saraga(DATA_HOME)

    def pitch_fn(track_id: str):
        return get_pitch_for_track(track_id, saraga)

    tonic_fn = None
    if args.annotated_tonic:
        def tonic_fn(track_id: str):  # noqa: F811
            return get_tonic_for_track(track_id, saraga)

    mode = "ANNOTATED tonic" if args.annotated_tonic else "ESTIMATED tonic"
    print(f"=== Building segment dataset ({mode}) ===")
    X, feature_names, records = build_segment_feature_dataset(
        tracks=TRACKS,
        get_pitch_fn=pitch_fn,
        segment_length_s=30.0,
        hop_s=20.0,
        min_duration_s=15.0,
        get_tonic_fn=tonic_fn,
    )

    failed_records = [r for r in records if r["failed"]]
    valid_records = [r for r in records if not r["failed"]]

    print(f"Valid segments: {len(valid_records)}")
    print(f"Failed segments: {len(failed_records)}")
    print(f"X shape: {X.shape}")
    print(f"# features: {len(feature_names)}")

    if feature_names:
        print("\nFirst 15 feature names:")
        for i, name in enumerate(feature_names[:15]):
            print(i, name)

    if failed_records:
        print("\nFirst 10 failed segments:")
        for rec in failed_records[:10]:
            print(rec)

    if X.shape[0] == 0:
        print("\nNo valid segments produced. Stop here and debug slicing / stage1.")
        return

    y = [r["raga_label"] for r in valid_records]
    groups = [r["track_id"] for r in valid_records]

    logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                random_state=42
            )),
        ]
    )

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    evaluate_model("Logistic Regression", logreg, X, y, groups, valid_records)
    evaluate_model("Random Forest", rf, X, y, groups, valid_records)


if __name__ == "__main__":
    main()

