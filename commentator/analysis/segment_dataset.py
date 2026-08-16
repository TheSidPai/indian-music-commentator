"""Segment-level dataset construction for raga classification baselines."""

from __future__ import annotations

from typing import Callable, Any, Protocol

import numpy as np
import pandas as pd

from .stage1_schema import build_stage1_schema
from .raga_features import extract_raga_features_from_stage1

class PitchContourLike(Protocol):
    times: Any



def plot_tsne_segments(
    X,
    labels,
    track_ids,
    segment_indices=None,
    out_path=None,
    annotate=False,
    max_annotations=80,
    random_state=0,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    track_ids = np.asarray(track_ids)

    if out_path is None:
        n_ragas = len(np.unique(labels))
        n_features = X.shape[1]
        out_path = f"tsne_segments_{n_ragas}raga_{n_features}feat.png"

    if segment_indices is None:
        segment_indices = np.arange(len(labels))
    else:
        segment_indices = np.asarray(segment_indices)

    if len(X) < 3:
        raise ValueError("Need at least 3 points for t-SNE.")

    X_scaled = StandardScaler().fit_transform(X)

    n = len(X_scaled)
    perplexity = min(30, max(5, n // 10))
    if perplexity >= n:
        perplexity = max(2, n - 1)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        perplexity=perplexity,
    )
    X_emb = tsne.fit_transform(X_scaled)

    colors = {
        "Raag_Bihag": "tab:blue",
        "Raag_Bhoopali": "tab:orange",
        "Raag_Kedar": "tab:green",
        "Raag_Abhogi": "tab:red",
        "Raag_Shree": "tab:purple",
        "Raag_Lalit": "tab:brown",
    }

    markers = {
        "27_Raag_Bihag": "o",
        "81_Raag_Bihag": "s",

        "83_Raag_Bhoopali": "^",
        "105_Raag_Bhoopali": "v",

        "8_Raag_Kedar": "D",
        "84_Raag_Kedar": "P",

        "20_Raag_Abhogi": "X",
        "44_Raag_Abhogi": "<",

        "0_Raag_Shree": ">",
        "37_Raag_Shree": "*",

        "10_Raag_Lalit": "h",
        "33_Raag_Lalit": "8",
        "104_Raga_Lalit_-_Khayal": "p",
    }

    plt.figure(figsize=(9, 7))

    unique_tracks = np.unique(track_ids)
    for tid in unique_tracks:
        mask = track_ids == tid
        raga = labels[mask][0]
        plt.scatter(
            X_emb[mask, 0],
            X_emb[mask, 1],
            c=colors.get(str(raga), "gray"),
            marker=markers.get(str(tid), "o"),
            s=28,
            alpha=0.75,
            edgecolors="none",
            label=str(tid),
        )

    if annotate:
        if len(X_emb) <= max_annotations:
            annotate_idx = np.arange(len(X_emb))
        else:
            rng = np.random.default_rng(random_state)
            annotate_idx = rng.choice(len(X_emb), size=max_annotations, replace=False)

        for i in annotate_idx:
            short_tid = str(track_ids[i]).split("_")[0]
            txt = f"{short_tid}:{int(segment_indices[i])}"
            plt.annotate(
                txt,
                (X_emb[i, 0], X_emb[i, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
            )

    plt.title(f"t-SNE of segment features (perplexity={perplexity})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(fontsize=7, loc="best", ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE plot to {out_path}")

def generate_track_segments(
    duration_s: float,
    segment_length_s: float = 30.0,
    hop_s: float = 30.0,
    min_duration_s: float = 15.0,
) -> list[tuple[float, float]]:
    """Generate time segments (start_s, end_s) for one track."""
    if duration_s <= 0:
        return []

    segments: list[tuple[float, float]] = []
    start_s = 0.0

    while start_s < duration_s:
        end_s = min(start_s + segment_length_s, duration_s)
        if (end_s - start_s) >= min_duration_s:
            segments.append((start_s, end_s))
        start_s += hop_s

    return segments


def slice_pitch_contour(pitch_obj: Any, start_s: float, end_s: float):
    """Return a time-sliced PitchContour object."""
    times = np.asarray(pitch_obj.times)
    mask = (times >= start_s) & (times < end_s)

    if not np.any(mask):
        raise ValueError(f"No pitch frames in segment [{start_s}, {end_s})")

    segment_times = times[mask] - start_s
    segment_frequencies = np.asarray(pitch_obj.frequencies)[mask]
    segment_voiced_mask = np.asarray(pitch_obj.voiced_mask)[mask]

    segment_confidence = None
    if getattr(pitch_obj, "confidence", None) is not None:
        segment_confidence = np.asarray(pitch_obj.confidence)[mask]

    pitch_cls = pitch_obj.__class__

    return pitch_cls(
        times=segment_times,
        frequencies=segment_frequencies,
        voiced_mask=segment_voiced_mask,
        confidence=segment_confidence,
        source=getattr(pitch_obj, "source", None),
        track_id=getattr(pitch_obj, "track_id", None),
    )


def build_segment_feature_dataset(
    tracks: list[dict],
    get_pitch_fn: Callable[[str], PitchContourLike],
    segment_length_s: float = 60.0,
    hop_s: float = 50.0,
    min_duration_s: float = 15.0,
) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Build a segment-level feature dataset.

    Each input track dict should contain:
        {
            "track_id": "...",
            "raga_label": "..."
        }

    Returns:
        X: feature matrix of valid segments
        feature_names: feature names
        records: metadata for each valid/failed segment
    """
    x_rows: list[np.ndarray] = []
    feature_names: list[str] = []
    records: list[dict] = []

    for track_info in tracks:
        track_id = track_info["track_id"]
        raga_label = track_info["raga_label"]

        pitch_obj = get_pitch_fn(track_id)

        duration_s = None
        if hasattr(pitch_obj, "times") and len(pitch_obj.times) > 0:
            duration_s = float(np.asarray(pitch_obj.times)[-1])

        if duration_s is None or duration_s <= 0:
            raise ValueError(f"Could not infer duration from pitch object for {track_id!r}")

        segments = generate_track_segments(
            duration_s=duration_s,
            segment_length_s=segment_length_s,
            hop_s=hop_s,
            min_duration_s=min_duration_s,
        )

        for segment_index, (start_s, end_s) in enumerate(segments):
            try:
                segment_pitch = slice_pitch_contour(pitch_obj, start_s, end_s)

                stage1 = build_stage1_schema(
                    segment_pitch,
                    raga_label=raga_label,
                    include_artifacts=False,
                )

                x, names, meta = extract_raga_features_from_stage1(stage1)
                x = np.asarray(x, dtype=float)


                if not feature_names:
                    feature_names = list(names)
                elif list(names) != feature_names:
                    raise ValueError(
                        f"Inconsistent feature names in track {track_id!r}, "
                        f"segment {segment_index}"
                    )

                x_rows.append(x)

                records.append(
                    {
                        "track_id": track_id,
                        "raga_label": raga_label,
                        "segment_index": segment_index,
                        "start_s": float(start_s),
                        "end_s": float(end_s),
                        "duration_s": float(end_s - start_s),
                        "failed": False,
                        "feature_meta": meta,
                    }
                )

            except Exception as exc:
                records.append(
                    {
                        "track_id": track_id,
                        "raga_label": raga_label,
                        "segment_index": segment_index,
                        "start_s": float(start_s),
                        "end_s": float(end_s),
                        "duration_s": float(end_s - start_s),
                        "failed": True,
                        "error": str(exc),
                    }
                )

    valid_records = [r for r in records if not r["failed"]]


    
    if not x_rows:
        return np.empty((0, 0), dtype=float), [], records

    X = np.vstack(x_rows)

    y = np.array([r["raga_label"] for r in valid_records])
    track_ids = np.array([r["track_id"] for r in valid_records])
    segment_indices = np.array([r["segment_index"] for r in valid_records])

    
    df = pd.DataFrame(X)
    df.insert(0, "raga_label", y)
    df.to_csv('key_segment_features_table.csv')

    n_ragas = len(np.unique(y))
    n_features = X.shape[1]
    tsne_out_path = (
        f"tsne_segments_{n_ragas}raga_{n_features}feat_"
        f"{int(segment_length_s)}s-{int(hop_s)}shop.png"
    )

    plot_tsne_segments(
        X,
        y,
        track_ids,
        segment_indices=segment_indices,
        out_path=tsne_out_path,
        annotate=False,
    )

    if len(valid_records) != X.shape[0]:
        raise RuntimeError(
            f"Mismatch between valid records ({len(valid_records)}) "
            f"and X rows ({X.shape[0]})."
        )

    return X, feature_names, records