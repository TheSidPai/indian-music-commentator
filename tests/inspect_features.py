from __future__ import annotations

import numpy as np
import pandas as pd

from commentator.io.loaders import initialize_saraga, get_pitch_for_track
from commentator.analysis.stage1_schema import build_stage1_schema
from commentator.analysis.raga_features import extract_raga_features_from_stage1

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(X, labels, track_ids, out_path=None):
    if out_path is None:
        n_ragas = len(np.unique(labels))
        n_features = X.shape[1]
        out_path = f"tsne_plot_{n_ragas}raga_{n_features}feat.png"

    tsne = TSNE(n_components=2, init="random", random_state=0, perplexity=3)
    X_emb = tsne.fit_transform(X)

    ragas = np.unique(labels)
    colors = {"Raag_Bihag": "tab:blue", "Raag_Yaman": "tab:orange", "Raag_Kedar": "tab:green"}

    plt.figure(figsize=(5, 4))
    for r in ragas:
        mask = (labels == r)
        plt.scatter(
            X_emb[mask, 0],
            X_emb[mask, 1],
            c=colors.get(r, "gray"),
            label=r,
        )

    for i, tid in enumerate(track_ids):
        plt.annotate(tid.split("_")[0], (X_emb[i, 0], X_emb[i, 1]), fontsize=8)

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved t-SNE plot to {out_path}")
    plt.close()

DATA_HOME = "/home/thesidpai/mir_projects/data"

def select_baseline_feature_subset(
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    removed:

        "confident_ratio",

        "min_relative_cents",
        "max_relative_cents",
        "median_relative_cents",
        "range_span_cents",

    
        "hist_entropy",
    """

    keep_names = [
        
        "swara_prop_Sa",
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

def build_feature_matrix():
    saraga = initialize_saraga(DATA_HOME)

    def pitch_fn(track_id: str):
        return get_pitch_for_track(track_id, saraga)

    tracks = [
        {"track_id": "27_Raag_Bihag", "raga_label": "Raag_Bihag"},
        {"track_id": "81_Raag_Bihag", "raga_label": "Raag_Bihag"},
        {"track_id": "12_Raag_Yaman", "raga_label": "Raag_Yaman"},
        {"track_id": "31_Raag_Yaman", "raga_label": "Raag_Yaman"},
        {"track_id": "8_Raag_Kedar", "raga_label": "Raag_Kedar"},
        {"track_id": "84_Raag_Kedar", "raga_label": "Raag_Kedar"},
    ]

    X_list = []
    y_list = []
    track_ids = []
    feature_names: list[str] = []

    for cfg in tracks:
        track_id = cfg["track_id"]
        label = cfg["raga_label"]

        pitch_obj = pitch_fn(track_id)
        stage1 = build_stage1_schema(
            pitch_obj,
            raga_label=label,
            include_artifacts=False,
        )
        x, names, meta = extract_raga_features_from_stage1(stage1)

        if not feature_names:
            feature_names = list(names)
        else:
            if list(names) != feature_names:
                raise ValueError(
                    f"Feature name mismatch for track {track_id}"
                )

        X_list.append(x)
        y_list.append(label)
        track_ids.append(track_id)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=object)
    track_ids = np.array(track_ids, dtype=object)

    return X, y, track_ids, feature_names


def main():
    X, y, track_ids, feature_names = build_feature_matrix()

    X, feature_names = select_baseline_feature_subset(X, feature_names)

    print("Reduced X shape:", X.shape)
    print("First 10 reduced feature names:", feature_names[:10])
    print("Track IDs:", list(track_ids))
    print("Labels:", list(y))
    print()

    # print("X shape:", X.shape)
    # print("Track IDs:", list(track_ids))
    # print("Labels:", list(y))
    # print("First 10 feature names:", feature_names[:10])
    # print()

    # Build a small table of key features for quick sanity check
    key_feature_names = [
        "swara_prop_Sa",
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
        "hist_entropy",

        "confident_ratio",

        "min_relative_cents",
        "max_relative_cents",
        "median_relative_cents",
        "range_span_cents",
        
    ]

    # Map name -> column index
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    cols_to_use = [name_to_idx[n] for n in key_feature_names if n in name_to_idx]
    used_names = [feature_names[i] for i in cols_to_use]

    df = pd.DataFrame(
        X[:, cols_to_use],
        columns=used_names,
        index=track_ids,
    )
    df.insert(0, "raga_label", y)
    df.to_csv('key_features_table.csv')
    print("=== Key feature snapshot per track ===")
    print(df)

    print()

    # Pairwise distances between tracks
    print("=== Pairwise Euclidean distances between track features ===")
    n = X.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            print(
                f"{track_ids[i]} ({y[i]})  <->  "
                f"{track_ids[j]} ({y[j]}):  distance = {d:.3f}"
            )

    plot_tsne(X, y, track_ids)
            


if __name__ == "__main__":
    main()