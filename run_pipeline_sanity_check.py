from commentator.analysis.stage1_schema import build_stage1_schema
from commentator.analysis.raga_features import extract_raga_features_from_stage1
from commentator.analysis.raga_baseline_knn import evaluate_knn_leave_one_track_out
from commentator.io.loaders import initialize_saraga, get_pitch_for_track


DATA_HOME = "/home/thesidpai/mir_projects/data"


def main():
    saraga = initialize_saraga(DATA_HOME)

    def pitch_fn(track_id: str):
        return get_pitch_for_track(track_id, saraga)

    print("=== Single-track sanity check ===")
    pitch_obj = pitch_fn("27_Raag_Bihag")

    stage1 = build_stage1_schema(
        pitch_obj,
        raga_label="Raag_Bihag",
        include_artifacts=False,
    )

    x, names, meta = extract_raga_features_from_stage1(stage1)

    print(meta)
    print(x.shape)
    print(names[:10])
    print(x[:10])
    print()

    print("=== Baseline evaluation ===")
    tracks = [
        {"track_id": "27_Raag_Bihag", "raga_label": "Raag_Bihag"},
        {"track_id": "81_Raag_Bihag", "raga_label": "Raag_Bihag"},

        {"track_id": "12_Raag_Yaman", "raga_label": "Raag_Yaman"},
        {"track_id": "31_Raag_Yaman", "raga_label": "Raag_Yaman"},

        {"track_id": "8_Raag_Kedar", "raga_label": "Raag_Kedar"},
        {"track_id": "84_Raag_Kedar", "raga_label": "Raag_Kedar"},
    ]

    result = evaluate_knn_leave_one_track_out(
        tracks=tracks,
        get_pitch_fn=pitch_fn,
        n_neighbors=3,
    )

    print("X shape:", result["X_shape"])
    print("Track accuracy:", result["track_accuracy"])
    print("Number of evaluated tracks:", result["n_tracks"])
    print("First 10 feature names:", result["feature_names"][:10])
    print()

    for row in result["track_results"]:
        print(row)


if __name__ == "__main__":
    main()