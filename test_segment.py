from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from commentator.io.loaders import initialize_saraga, get_pitch_for_track
from commentator.analysis.segment_dataset import build_segment_feature_dataset


DATA_HOME = "/home/thesidpai/mir_projects/data"  # change if needed

TRACKS = [
    {"track_id": "27_Raag_Bihag", "raga_label": "Raag_Bihag"},
    {"track_id": "81_Raag_Bihag", "raga_label": "Raag_Bihag"},
    {"track_id": "8_Raag_Kedar",  "raga_label": "Raag_Kedar"},
    {"track_id": "84_Raag_Kedar", "raga_label": "Raag_Kedar"},
    {"track_id": "83_Raag_Bhoopali",  "raga_label": "Raag_Bhoopali"},
    {"track_id": "105_Raag_Bhoopali", "raga_label": "Raag_Bhoopali"},
]

def inspect_kedar_confusions(X, y, records, n_neighbors=5, top_k_segments=10):
    """
    For each Kedar track, find segments whose nearest neighbours are mostly Bhoopali,
    and print their time ranges so you can listen to them.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    labels = np.array(y)
    recs = list(records)

    # Indices for each raga
    idx_bhoop = np.where(labels == "Raag_Bhoopali")[0]
    idx_kedar = np.where(labels == "Raag_Kedar")[0]

    if len(idx_bhoop) == 0 or len(idx_kedar) == 0:
        print("Need both Bhoopali and Kedar segments for this analysis.")
        return

    # Fit NN on ALL segments
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X)

    print("\n=== Kedar segments with Bhoopali-like neighbourhoods ===")

    for kedar_track in sorted({recs[i]["track_id"] for i in idx_kedar}):
        kedar_idx = [i for i in idx_kedar if recs[i]["track_id"] == kedar_track]
        if not kedar_idx:
            continue

        scores = []

        for i in kedar_idx:
            x_i = X[i : i + 1]
            dists, neigh_idx = nn.kneighbors(x_i, return_distance=True)

            # drop self from neighbours
            neigh_idx = neigh_idx[0]
            neigh_idx = [j for j in neigh_idx if j != i][:n_neighbors]

            neigh_labels = labels[neigh_idx]
            bhoop_count = np.sum(neigh_labels == "Raag_Bhoopali")
            bihag_count = np.sum(neigh_labels == "Raag_Bihag")
            kedar_count = np.sum(neigh_labels == "Raag_Kedar")

            # score = fraction of Bhoopali neighbours
            frac_bhoop = bhoop_count / max(1, len(neigh_idx))

            scores.append(
                {
                    "index": i,
                    "frac_bhoop": frac_bhoop,
                    "bhoop": int(bhoop_count),
                    "bihag": int(bihag_count),
                    "kedar": int(kedar_count),
                }
            )

        scores.sort(key=lambda s: s["frac_bhoop"], reverse=True)
        top = scores[:top_k_segments]

        print(f"\nKedar track: {kedar_track}")
        for s in top:
            rec = recs[s["index"]]
            print(
                f"  seg {rec['segment_index']:3d} | "
                f"{rec['start_s']:7.1f}–{rec['end_s']:7.1f}s | "
                f"Bhoop NN frac={s['frac_bhoop']:.2f} "
                f"(Bhoop={s['bhoop']}, Bihag={s['bihag']}, Kedar={s['kedar']})"
            )

def majority_vote(values: list[str]) -> str:
    if not values:
        return "UNKNOWN"
    return Counter(values).most_common(1)[0][0]


def main() -> None:
    saraga = initialize_saraga(DATA_HOME)

    def pitch_fn(track_id: str):
        return get_pitch_for_track(track_id, saraga)

    print("=== Building segment dataset ===")
    X, feature_names, records = build_segment_feature_dataset(
        tracks=TRACKS,
        get_pitch_fn=pitch_fn,
        segment_length_s=30.0,
        hop_s=20.0,
        min_duration_s=15.0,
    )

    failed_records = [r for r in records if r["failed"]]
    valid_records = [r for r in records if not r["failed"]]

    print(f"Valid segments:  {len(valid_records)}")
    print(f"Failed segments: {len(failed_records)}")
    print(f"X shape:         {X.shape}")
    print(f"# features:      {len(feature_names)}")

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

    logo = LeaveOneGroupOut()

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=3, metric="euclidean")),
        ]
    )

    track_results = []
    segment_correct = []

    print("\n=== Leave-one-track-out results ===")

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        test_records = [valid_records[i] for i in test_idx]
        held_out_track = test_records[0]["track_id"]
        true_label = test_records[0]["raga_label"]

        clf.fit(X_train, y_train)
        seg_preds = [str(p) for p in clf.predict(X_test)]

        for pred, truth in zip(seg_preds, y_test):
            segment_correct.append(pred == truth)

        vote_counts = {str(k): v for k, v in Counter(seg_preds).items()}
        track_pred = majority_vote(seg_preds)
        track_correct = track_pred == true_label
        seg_acc = sum(p == t for p, t in zip(seg_preds, y_test)) / len(y_test)

        result = {
            "track_id": held_out_track,
            "true_label": str(true_label),
            "track_pred": str(track_pred),
            "track_correct": track_correct,
            "n_segments": len(test_idx),
            "segment_accuracy": seg_acc,
            "vote_counts": vote_counts,
        }
        track_results.append(result)

        print(f"\nHeld-out track: {held_out_track}")
        print(f"  True label:      {true_label}")
        print(f"  Pred label:      {track_pred}")
        print(f"  Track correct:   {track_correct}")
        print(f"  Num segments:    {len(test_idx)}")
        print(f"  Segment acc:     {seg_acc:.3f}")
        print(f"  Vote counts:     {vote_counts}")

    track_acc = sum(r["track_correct"] for r in track_results) / len(track_results)
    seg_acc = sum(segment_correct) / len(segment_correct)

    print("\n=== Summary ===")
    print(f"Track-level accuracy:   {track_acc:.3f}")
    print(f"Segment-level accuracy: {seg_acc:.3f}")

    print("\n=== Full track results ===")
    for r in track_results:
        print(r)

    inspect_kedar_confusions(X, y, valid_records, n_neighbors=5, top_k_segments=10)


if __name__ == "__main__":
    main()