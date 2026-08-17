"""Segment-level dataset construction for raga classification baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Any, Protocol

import numpy as np
import pandas as pd

from .stage1_schema import build_stage1_schema
from .raga_features import extract_raga_features_from_stage1

# Resolved relative to this file (not the caller's cwd), so generated
# artifacts always land in <repo root>/outputs/ regardless of where a
# script that imports this module is run from.
OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"

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
    max_points=4000,
):
    """Plot a t-SNE of segment features.

    max_points:
        t-SNE is O(n log n) at best and the scatter becomes an unreadable
        blob well before that matters, so larger inputs are randomly
        subsampled (stratified by track) down to this many points. Saraga's
        1809-segment runs are unaffected.

    Legend/colour behaviour adapts to the dataset: with few tracks (Saraga)
    each track gets its own marker and legend entry, which is useful for
    spotting a single bad recording. With many tracks (HMD's 30 ragas x 10)
    that legend would be useless, so points are coloured and labelled by
    raga instead.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    track_ids = np.asarray(track_ids)

    if segment_indices is not None:
        segment_indices = np.asarray(segment_indices)

    if max_points is not None and len(X) > max_points:
        rng = np.random.default_rng(random_state)
        # Sample proportionally within each track so no recording drops out.
        keep = []
        for tid in np.unique(track_ids):
            idx = np.flatnonzero(track_ids == tid)
            n_keep = max(1, int(round(len(idx) * max_points / len(X))))
            keep.append(rng.choice(idx, size=min(n_keep, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        print(f"t-SNE: subsampling {len(X)} -> {len(keep)} points for plotting")
        X = X[keep]
        labels = labels[keep]
        track_ids = track_ids[keep]
        if segment_indices is not None:
            segment_indices = segment_indices[keep]

    if out_path is None:
        n_ragas = len(np.unique(labels))
        n_features = X.shape[1]
        out_path = OUTPUTS_DIR / f"tsne_segments_{n_ragas}raga_{n_features}feat.png"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

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

    # Any raga not in the hand-picked Saraga palette (i.e. every HMD raga)
    # gets a deterministic colour from a colormap, so plots are not all grey.
    unique_labels = sorted({str(v) for v in labels})
    unassigned = [lbl for lbl in unique_labels if lbl not in colors]
    if unassigned:
        cmap = plt.get_cmap("hsv", len(unassigned) + 1)
        for i, lbl in enumerate(unassigned):
            colors[lbl] = cmap(i)

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
    # One legend entry per track is informative for Saraga's 13 recordings but
    # useless for HMD's 300, so switch to per-raga grouping past a threshold.
    legend_by_track = len(unique_tracks) <= 15

    if legend_by_track:
        for tid in unique_tracks:
            mask = track_ids == tid
            raga = labels[mask][0]
            plt.scatter(
                X_emb[mask, 0],
                X_emb[mask, 1],
                c=[colors.get(str(raga), "gray")],
                marker=markers.get(str(tid), "o"),
                s=28,
                alpha=0.75,
                edgecolors="none",
                label=str(tid),
            )
    else:
        for lbl in unique_labels:
            mask = labels.astype(str) == lbl
            plt.scatter(
                X_emb[mask, 0],
                X_emb[mask, 1],
                c=[colors.get(lbl, "gray")],
                marker="o",
                s=14,
                alpha=0.65,
                edgecolors="none",
                label=lbl,
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

    n_ragas_plotted = len(unique_labels)
    plt.title(
        f"t-SNE of segment features "
        f"({len(X_emb)} segments, {n_ragas_plotted} ragas, perplexity={perplexity})"
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if legend_by_track:
        plt.legend(fontsize=7, loc="best", ncol=3)
    else:
        plt.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
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
    get_tonic_fn: Callable[[str], float | None] | None = None,
    run_tag: str = "",
) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Build a segment-level feature dataset.

    Each input track dict should contain:
        {
            "track_id": "...",
            "raga_label": "..."
        }

    get_tonic_fn:
        Optional track_id -> tonic Hz lookup (e.g. a dataset adapter's
        get_tonic). When given, every segment of that track is analyzed
        against the supplied tonic instead of estimating one per window.
        Both Saraga and CompMusic HMD ship annotated tonics; per-segment
        estimation is unreliable (see tests/run_tonic_validation.py).

    run_tag:
        Optional discriminator inserted into the generated CSV/PNG filenames
        (e.g. "_compmusic-hmd_pilot1"). Runs that differ in dataset or track
        subset can otherwise collide: the filenames encode raga count,
        feature count and window, none of which distinguish two datasets
        that happen to share them. Empty by default, so existing Saraga
        artifact names are unchanged.

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
        track_tonic_hz = get_tonic_fn(track_id) if get_tonic_fn is not None else None

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
                    tonic_hz=track_tonic_hz,
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

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # An annotated-tonic run has the same raga/feature/window counts as an
    # estimated-tonic one, so without this tag the two would overwrite each
    # other's artifacts.
    tonic_tag = "_annotated-tonic" if get_tonic_fn is not None else ""

    # track_id must be exported, not just used in-memory: without it, consumers
    # of this CSV cannot tell which rows came from the same recording, and any
    # random train/test split will scatter one recording across both sides.
    # Segments are overlapping windows of a single performance, so that leaks
    # near-duplicate rows into the test set and inflates accuracy badly (this
    # is exactly how the retired 0.9051 figure arose -- see the 2026-08-16
    # entry in docs/experiments/2026-06-raga-baseline-log.md).
    #
    # It is a grouping key, NOT a feature. Kept as a string so that consumers
    # selecting numeric columns as features cannot pick it up by accident;
    # segment_index is deliberately not exported for the same reason.
    df = pd.DataFrame(X)
    df.insert(0, "track_id", track_ids)
    df.insert(0, "raga_label", y)
    df.to_csv(OUTPUTS_DIR / f"key_segment_features_table{run_tag}{tonic_tag}.csv")

    n_ragas = len(np.unique(y))
    n_features = X.shape[1]
    tsne_out_path = OUTPUTS_DIR / (
        f"tsne_segments{run_tag}_{n_ragas}raga_{n_features}feat_"
        f"{int(segment_length_s)}s-{int(hop_s)}shop{tonic_tag}.png"
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