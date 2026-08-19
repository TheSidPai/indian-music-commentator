from __future__ import annotations

import argparse
import sys
import datetime
import json
import subprocess
from pathlib import Path
import time
import collections
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
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


def select_tracks(all_tracks, ragas=None, per_raga=None, seed=42):
    """Subset a dataset's track list by raga and/or recordings-per-raga.

    Selection is deterministic given a seed: recordings are sorted by
    track_id before sampling, so the same subset is reproducible across runs.
    """
    import random

    if ragas:
        available = {t["raga_label"] for t in all_tracks}
        # Exact (case-insensitive) match wins; substring is only a fallback for
        # terms that match nothing exactly. Otherwise "Bhairav" would silently
        # also pull in "Ahira bhairav".
        keep: set[str] = set()
        for term in ragas:
            exact = {a for a in available if a.lower() == term.lower()}
            if exact:
                keep |= exact
                continue
            partial = {a for a in available if term.lower() in a.lower()}
            if partial:
                print(f"  note: '{term}' matched by substring -> {sorted(partial)}")
                keep |= partial
            else:
                print(f"  warning: '{term}' matched no raga")
        all_tracks = [t for t in all_tracks if t["raga_label"] in keep]
        print(f"Raga filter matched {len(keep)} raga(s): {sorted(keep)}")

    by_raga: dict[str, list[dict]] = defaultdict(list)
    for t in all_tracks:
        by_raga[t["raga_label"]].append(t)

    selected = []
    rng = random.Random(seed)
    for raga in sorted(by_raga):
        recordings = sorted(by_raga[raga], key=lambda t: t["track_id"])
        if per_raga is not None and len(recordings) > per_raga:
            recordings = sorted(rng.sample(recordings, per_raga),
                                key=lambda t: t["track_id"])
        selected.extend(recordings)

    return [{"track_id": t["track_id"], "raga_label": t["raga_label"]}
            for t in selected]


def majority_vote(values: list[str]) -> str:
    if not values:
        return "UNKNOWN"
    return Counter(values).most_common(1)[0][0]


def load_run_manifest(run_dir: Path) -> dict:
    """Read a saved run's manifest so an eval-only pass inherits its identity.

    Dataset, tonic mode and window are properties of the extraction, not of
    the evaluation, so they come from the manifest rather than being retyped
    on the command line where they could silently disagree with the table.
    """
    path = run_dir / "manifest.json"
    if not path.exists():
        raise SystemExit(f"no manifest.json in {run_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run_features(run_dir: Path, drop_features=None):
    """Load a saved run's features.csv.gz instead of re-extracting it.

    `apply_feature_subset` is pure column masking, and both imputation and
    scaling are fitted per column, so evaluating a subset of the saved table
    is numerically identical to re-extracting with those features removed.

    Nothing already in the run directory is rewritten -- features.csv.gz,
    tsne.png and manifest.json are read-only here, and only eval/ gains
    files. Row order is the extraction order preserved by to_csv, so the
    CV folds (which depend only on labels, groups and seed) are identical
    to the ones the original run used.
    """
    import pandas as pd

    csv_path = run_dir / "features.csv.gz"
    if not csv_path.exists():
        raise SystemExit(f"no features.csv.gz in {run_dir}")

    df = pd.read_csv(csv_path, dtype={"track_id": str, "raga_label": str})
    for required in ("raga_label", "track_id"):
        if required not in df.columns:
            raise SystemExit(f"{csv_path} has no {required!r} column")

    # track_id and raga_label are keys, never features -- see the comment in
    # segment_dataset.build_segment_feature_dataset.
    feature_names = [c for c in df.columns if c not in ("raga_label", "track_id")]

    drop = list(drop_features or [])
    unknown = [n for n in drop if n not in feature_names]
    if unknown:
        raise SystemExit(
            "--drop-features names absent from the table: " + ", ".join(unknown)
            + f"\n({len(feature_names)} feature columns available)"
        )
    dropped = set(drop)
    kept = [c for c in feature_names if c not in dropped]
    if not kept:
        raise SystemExit("--drop-features would remove every feature")

    X = df[kept].to_numpy(dtype=float)
    records = [
        {"track_id": tid, "raga_label": lbl, "failed": False}
        for tid, lbl in zip(df["track_id"], df["raga_label"])
    ]
    return X, kept, records, sorted(dropped)


OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
INDEX_PATH = OUTPUTS_DIR / "INDEX.md"

INDEX_HEADER = [
    "# Experiment index",
    "",
    "One row per evaluation. Appended automatically by `run_segment_lr_rf.py`.",
    "Feature tables, plots and manifests live in `outputs/runs/<run>/`;",
    "every evaluation of a run's features is in that run's `eval/`.",
    "",
    "| run | dataset | ragas | tracks | feats | tonic | grouping | CV | model | track | segment | xchance |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]


def run_id_for(args, n_ragas, n_tracks):
    """Directory name for one feature extraction.

    Identity = everything that changes the numbers inside features.csv.gz:
    dataset, track subset, tonic mode, window. Things that only change how
    those features are *evaluated* (grouping, classifier, CV) do not get a new
    directory -- they become files in the run's eval/.
    """
    if args.run_tag:
        return args.run_tag.lstrip("_")
    date = datetime.date.today().isoformat()
    tonic = "annotated" if args.annotated_tonic else "estimated"
    subset = f"{n_ragas}raga-{n_tracks}track"
    win = f"{int(args.segment_length)}s-{int(args.hop)}hop"
    return f"{date}_{args.dataset}_{subset}_{win}_{tonic}"


def git_commit() -> str | None:
    """Short SHA of HEAD, or None outside a repo / on any failure."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent
                              ).stdout.strip() or None
    except Exception:
        return None


def write_manifest(run_dir, args, argv, X, feature_names, n_tracks, n_ragas,
                   build_seconds, tracks_per_raga, segments_per_raga):
    """Single source of truth for what produced this run's features."""
    commit = git_commit()

    manifest = {
        "run_id": run_dir.name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "command": " ".join(argv),
        "dataset": args.dataset,
        "tonic": "annotated" if args.annotated_tonic else "estimated",
        "n_tracks": n_tracks,
        "n_ragas": n_ragas,
        "n_segments": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": list(feature_names),
        "segment_length_s": args.segment_length,
        "hop_s": args.hop,
        "min_duration_s": 15.0,
        "build_seconds": round(build_seconds, 1),
        "tracks_per_raga": dict(sorted(tracks_per_raga.items())),
        "segments_per_raga": dict(sorted(segments_per_raga.items())),
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  wrote {run_dir / 'manifest.json'}")


def append_to_index(run_dir, args, run, n_tracks, n_ragas, n_features,
                    feats_label=None):
    """Append this evaluation's headline numbers to outputs/INDEX.md.

    `feats_label` carries the feature-set name into the 'feats' column so
    that several evaluations of one table (an ablation series, say) stay
    distinguishable in the index without changing its column layout.
    """
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text("\n".join(INDEX_HEADER) + "\n", encoding="utf-8")
    tonic = "ann" if args.annotated_tonic else "est"
    rows = []
    for r in run["results"]:
        rows.append(
            f"| {run_dir.name} | {args.dataset} | {n_ragas} | {n_tracks} | "
            f"{feats_label or n_features} | {tonic} | {run['grouping']} | {run['cv']} | "
            f"{r['model']} | {r['track_accuracy']:.4f} | "
            f"{r['segment_accuracy']:.4f} | {r['segment_accuracy'] / r['chance']:.1f}x |"
        )
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"  appended {len(rows)} row(s) to {INDEX_PATH}")


def write_results(args, grouping, run, X, n_tracks, n_classes, build_seconds,
                  tracks_per_raga, segments_per_raga, run_dir,
                  feature_set="71feat-full", feature_names=None,
                  features_dropped=None, from_run=False):
    """Persist one grouping's results as JSON + a readable report.

    Written into the run's eval/ so several evaluations of the same feature
    table stay distinct without duplicating the table itself.

    Each file records *which* features it used, not just how many. The
    filename is only a label: a count cannot be inverted back to a set, so
    without `features_used` / `features_dropped` an evaluation of a feature
    subset is not reproducible from its own artifacts. Run directories carry
    identity in "directory name + manifest"; eval/ needs the same standard.
    """
    from sklearn.metrics import classification_report
    import textwrap

    outdir = run_dir / "eval"
    outdir.mkdir(parents=True, exist_ok=True)
    tonic_tag = "annotated" if args.annotated_tonic else "estimated"
    stem = f"{feature_set}_by-{grouping}_lr-rf"

    # Same rule as the run directory itself: never silently replace a saved
    # result. This project has lost results to overwriting twice.
    if (outdir / f"{stem}.json").exists() and not args.overwrite:
        raise SystemExit(
            f"eval result already exists:\n  {outdir / stem}.json\n"
            f"Pass --feature-set to name this evaluation differently, "
            f"or --overwrite to replace it."
        )

    feature_names = list(feature_names or [])
    features_dropped = list(features_dropped or [])
    n_in_table = int(X.shape[1]) + len(features_dropped)

    summary = {
        "dataset": args.dataset,
        "tonic": tonic_tag,
        "grouping_unit": grouping,
        "cv": run["cv"],
        "n_tracks": n_tracks,
        "n_ragas": n_classes,
        "n_segments": int(X.shape[0]),
        # What distinguishes this evaluation from others of the same table.
        "feature_set": feature_set,
        "n_features": int(X.shape[1]),
        "n_features_in_table": n_in_table,
        "features_used": feature_names,
        "features_dropped": features_dropped,
        "evaluated_from": ("saved features.csv.gz (eval-only pass)" if from_run
                           else "features extracted by this invocation"),
        "eval_command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "n_cv_groups": run["n_groups"],
        "groups_per_raga": run["groups_per_raga"],
        "tracks_per_raga": dict(sorted(tracks_per_raga.items())),
        "segments_per_raga": dict(sorted(segments_per_raga.items())),
        "segment_length_s": args.segment_length,
        "hop_s": args.hop,
        # null on an eval-only pass: no extraction happened here, so 0.0 would
        # read as "extraction was instant" rather than "not measured".
        "extraction_build_seconds": (None if from_run or not build_seconds
                                     else round(build_seconds, 1)),
        "chance": run["results"][0]["chance"],
        "models": {
            r["model"]: {
                "track_accuracy": r["track_accuracy"],
                "segment_accuracy": r["segment_accuracy"],
                "segment_x_chance": r["segment_accuracy"] / r["chance"],
            }
            for r in run["results"]
        },
    }
    with open(outdir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(outdir / f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write(f"{args.dataset}  |  {tonic_tag} tonic  |  grouped by {grouping}\n")
        f.write("=" * 72 + "\n\n")

        # What makes this evaluation different from its siblings, up front.
        if features_dropped:
            f.write(f"Feature set: {feature_set}  --  {X.shape[1]} of "
                    f"{n_in_table} columns, {len(features_dropped)} dropped\n")
        else:
            f.write(f"Feature set: {feature_set}  --  all {n_in_table} columns\n")
        f.write(f"CV: {run['cv']}  ({run['n_groups']} groups over {n_tracks} tracks)\n")
        f.write(f"Ragas: {n_classes} | segments: {X.shape[0]} | features: {X.shape[1]}\n")
        f.write(f"Window: {args.segment_length}s / {args.hop}s hop\n")
        f.write(f"Chance: {summary['chance']:.4f}\n")
        f.write(f"Command: {summary['eval_command']}\n")
        if summary["git_commit"]:
            f.write(f"Commit: {summary['git_commit']}\n")
        f.write("\n")

        # A count cannot be inverted back to a set, so name the columns.
        if features_dropped:
            f.write(f"Features dropped ({len(features_dropped)})\n")
            f.write("-" * 30 + "\n")
            f.write(textwrap.fill(", ".join(features_dropped), width=72,
                                  initial_indent="  ", subsequent_indent="  "))
            f.write("\n\n")
        if feature_names:
            f.write(f"Features used ({len(feature_names)})\n")
            f.write("-" * 30 + "\n")
            f.write(textwrap.fill(", ".join(feature_names), width=72,
                                  initial_indent="  ", subsequent_indent="  "))
            f.write("\n\n")

        f.write("Accuracy\n--------\n")
        for r in run["results"]:
            f.write(f"  {r['model']:<22} track {r['track_accuracy']:.4f}   "
                    f"segment {r['segment_accuracy']:.4f}   "
                    f"({r['segment_accuracy'] / r['chance']:.1f}x chance)\n")

        f.write("\nPer-raga structure (bias check)\n")
        f.write("-------------------------------\n")
        f.write(f"{'raga':<22}{'tracks':>7}{'segments':>10}{'cv_groups':>11}\n")
        for raga in sorted(tracks_per_raga):
            f.write(f"{raga:<22}{tracks_per_raga[raga]:>7}"
                    f"{segments_per_raga.get(raga, 0):>10}"
                    f"{run['groups_per_raga'].get(raga, 0):>11}\n")

        for r in run["results"]:
            f.write(f"\n\nPer-class report -- {r['model']}\n")
            f.write("-" * 60 + "\n")
            f.write(classification_report(r["all_y_true"], r["all_y_pred"],
                                          digits=4, zero_division=0))
            wrong = [t for t in r["track_results"] if not t["track_correct"]]
            f.write(f"\nMisclassified tracks ({len(wrong)}/{len(r['track_results'])}):\n")
            for t in sorted(wrong, key=lambda t: t["true_label"]):
                f.write(f"  {t['true_label']:<22} -> {t['track_pred']:<22} "
                        f"({t['n_segments']} segs)  {t['track_id']}\n")

    print(f"  wrote {outdir / stem}.json / .txt")


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


def evaluate_model(model_name, clf, X, y, groups, valid_records, splitter,
                   per_fold_detail=True, show_confusion=True):
    """Evaluate under a group-aware CV splitter.

    Works for both LeaveOneGroupOut (one track per test fold) and
    StratifiedGroupKFold (many tracks per test fold). Track-level predictions
    are therefore aggregated per track_id rather than assuming the fold *is*
    a single track: with k-fold, one fold holds out many recordings at once.
    Every track still appears in exactly one test fold under both splitters,
    so each track is voted on exactly once.
    """
    segment_correct = []
    all_y_true = []
    all_y_pred = []
    votes_by_track: dict[str, list[str]] = defaultdict(list)
    truth_by_track: dict[str, str] = {}

    print(f"\n\n===== {model_name} =====")
    print(f"=== {type(splitter).__name__} results ===")

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups)):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = [y[i] for i in train_idx]
        y_test = [str(y[i]) for i in test_idx]

        clf.fit(X_train, y_train)
        seg_preds = [str(p) for p in clf.predict(X_test)]

        all_y_true.extend(y_test)
        all_y_pred.extend(seg_preds)
        segment_correct.extend(p == t for p, t in zip(seg_preds, y_test))

        for i, pred in zip(test_idx, seg_preds):
            tid = valid_records[i]["track_id"]
            votes_by_track[tid].append(pred)
            truth_by_track[tid] = str(valid_records[i]["raga_label"])

        fold_tracks = {valid_records[i]["track_id"] for i in test_idx}
        fold_acc = sum(p == t for p, t in zip(seg_preds, y_test)) / len(y_test)
        if per_fold_detail:
            print(f"\nFold {fold_i}: {len(fold_tracks)} held-out track(s), "
                  f"{len(test_idx)} segments, segment acc {fold_acc:.3f}")
            if len(fold_tracks) <= 3:
                for tid in sorted(fold_tracks):
                    print(f"  {tid}: true={truth_by_track[tid]} "
                          f"pred={majority_vote(votes_by_track[tid])}")

    track_results = []
    for tid, votes in votes_by_track.items():
        pred = majority_vote(votes)
        track_results.append({
            "track_id": tid,
            "true_label": truth_by_track[tid],
            "track_pred": pred,
            "track_correct": pred == truth_by_track[tid],
            "n_segments": len(votes),
        })

    track_acc = sum(r["track_correct"] for r in track_results) / len(track_results)
    seg_acc = sum(segment_correct) / len(segment_correct)
    n_classes = len(set(all_y_true))
    chance = 1.0 / n_classes

    print("\n=== Summary ===")
    print(f"Tracks evaluated: {len(track_results)} | classes: {n_classes}")
    print(f"Track-level accuracy:   {track_acc:.4f}  ({track_acc / chance:.1f}x chance)")
    print(f"Segment-level accuracy: {seg_acc:.4f}  ({seg_acc / chance:.1f}x chance)")
    print(f"Chance baseline:        {chance:.4f}")

    if show_confusion:
        print_segment_confusion(all_y_true, all_y_pred)

    print("\n=== Classification report ===")
    print(classification_report(all_y_true, all_y_pred, digits=4, zero_division=0))

    return {
        "model": model_name,
        "track_results": track_results,
        "track_accuracy": track_acc,
        "segment_accuracy": seg_acc,
        "chance": chance,
        "n_classes": n_classes,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotated-tonic",
        action="store_true",
        help="use the dataset's annotated tonic instead of estimating one per "
             "segment (see tests/run_tonic_validation.py for why this matters)",
    )
    parser.add_argument("--dataset", choices=["saraga", "compmusic_hmd"],
                        default="saraga")
    parser.add_argument("--ragas", nargs="*", default=None,
                        help="restrict to these raga labels (compmusic_hmd only)")
    parser.add_argument("--per-raga", type=int, default=None,
                        help="cap recordings per raga (compmusic_hmd only)")
    parser.add_argument("--cv", choices=["logo", "sgkf"], default="logo",
                        help="logo = leave-one-track-out; sgkf = StratifiedGroupKFold")
    parser.add_argument("--group-by", nargs="+", choices=["track", "album"],
                        default=["track"],
                        help="CV grouping unit. 'track' prevents segment leakage; "
                             "'album' additionally prevents the album/session effect, "
                             "where the model recognises a shared recording session "
                             "(same artist/tanpura/room) rather than the raga. "
                             "compmusic_hmd only.")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--segment-length", type=float, default=30.0)
    parser.add_argument("--hop", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default="",
                        help="explicit name for this run's directory under "
                             "outputs/runs/. If omitted, one is derived from date, "
                             "dataset, subset, window and tonic mode.")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow writing into a run directory that already exists")
    parser.add_argument("--skip-classification", action="store_true",
                        help="extract features and write artifacts, then stop")
    parser.add_argument("--from-run", default=None, metavar="RUN_ID",
                        help="evaluate an existing run's features.csv.gz instead "
                             "of re-extracting. Dataset, tonic mode and window are "
                             "read from that run's manifest. Only the run's eval/ "
                             "is written to.")
    parser.add_argument("--drop-features", nargs="+", default=None, metavar="NAME",
                        help="feature columns to exclude (--from-run only). Pure "
                             "column masking, so results are identical to "
                             "re-extracting without those features.")
    parser.add_argument("--feature-set", default=None, metavar="NAME",
                        help="label for this evaluation's output files, e.g. "
                             "'62feat-passA'. Defaults to '<n>feat-full'.")
    args = parser.parse_args()

    if args.drop_features and not args.from_run:
        parser.error("--drop-features requires --from-run")

    from_run_dir = None
    inherited_build_seconds = 0.0
    if args.from_run:
        from_run_dir = RUNS_DIR / args.from_run.lstrip("_")
        if not from_run_dir.is_dir():
            parser.error(f"no such run directory: {from_run_dir}")
        manifest = load_run_manifest(from_run_dir)
        # The extraction's identity comes from the manifest, not the command
        # line, so an eval-only pass cannot mislabel the table it is reading.
        args.dataset = manifest["dataset"]
        args.annotated_tonic = str(manifest.get("tonic", "")).startswith("annotated")
        args.segment_length = manifest.get("segment_length_s", args.segment_length)
        args.hop = manifest.get("hop_s", args.hop)
        inherited_build_seconds = float(manifest.get("build_seconds") or 0.0)
        print(f"=== Evaluating saved run: {from_run_dir.name} ===")
        print(f"  from manifest: dataset={args.dataset}  "
              f"tonic={'annotated' if args.annotated_tonic else 'estimated'}  "
              f"window={args.segment_length}s/{args.hop}s hop")

    if args.dataset == "saraga":
        saraga = initialize_saraga(DATA_HOME)

        def pitch_fn(track_id: str):
            return get_pitch_for_track(track_id, saraga)

        def annotated_tonic_fn(track_id: str):
            return get_tonic_for_track(track_id, saraga)

        tracks = list(TRACKS)
        if "album" in args.group_by:
            parser.error("--group-by album is only available for compmusic_hmd "
                         "(Saraga exposes no album metadata)")
        group_key_of = {"track": {t["track_id"]: t["track_id"] for t in tracks}}
    else:
        from commentator.io.compmusic import CompMusicHindustani

        hmd = CompMusicHindustani()
        pitch_fn = hmd.get_pitch
        annotated_tonic_fn = hmd.get_tonic
        all_hmd = hmd.list_tracks()
        tracks = select_tracks(all_hmd, args.ragas, args.per_raga, args.seed)

        # A "session" is one artist's one album. Recordings sharing it were
        # captured together (same tanpura, tonic, room), so holding out a track
        # while training on its album siblings lets the model recognise the
        # session rather than the raga.
        group_key_of = {
            "track": {t["track_id"]: t["track_id"] for t in all_hmd},
            "album": {
                t["track_id"]: f"{t['artist']}||{t['pitch_path'].parts[-2]}"
                for t in all_hmd
            },
        }

    tonic_fn = annotated_tonic_fn if args.annotated_tonic else None

    mode = "ANNOTATED tonic" if args.annotated_tonic else "ESTIMATED tonic"
    n_ragas_in = len({t["raga_label"] for t in tracks})

    if from_run_dir is not None:
        run_dir = from_run_dir
        X, feature_names, records, features_dropped = load_run_features(
            from_run_dir, args.drop_features
        )
        build_seconds = inherited_build_seconds
        valid_records = records
        dropped = features_dropped

        print(f"\n=== Loaded saved features ===")
        print(f"  {run_dir / 'features.csv.gz'}")
        print(f"  segments: {X.shape[0]}  features: {X.shape[1]}"
              f"  tracks: {len({r['track_id'] for r in valid_records})}")
        if dropped:
            print(f"  dropped {len(dropped)} feature(s):")
            for name in dropped:
                print(f"    - {name}")
        else:
            print("  dropped 0 features (control pass: full saved table)")
    else:
        run_dir = RUNS_DIR / run_id_for(args, n_ragas_in, len(tracks))
        if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
            parser.error(
                f"run directory already exists and is not empty:\n  {run_dir}\n"
                f"Pass --run-tag to name this run differently, or --overwrite to replace it."
            )

        print(f"=== Building segment dataset ===")
        print(f"  dataset : {args.dataset}")
        print(f"  tonic   : {mode}")
        print(f"  tracks  : {len(tracks)} across {n_ragas_in} ragas")
        print(f"  window  : {args.segment_length}s / {args.hop}s hop")
        print(f"  run dir : outputs/runs/{run_dir.name}/")

        t0 = time.time()
        X, feature_names, records = build_segment_feature_dataset(
            tracks=tracks,
            get_pitch_fn=pitch_fn,
            segment_length_s=args.segment_length,
            hop_s=args.hop,
            min_duration_s=15.0,
            get_tonic_fn=tonic_fn,
            run_dir=run_dir,
        )
        build_seconds = time.time() - t0
        features_dropped = []  # extraction path applies DROP_NAMES upstream

        failed_records = [r for r in records if r["failed"]]
        valid_records = [r for r in records if not r["failed"]]

        print(f"\n=== Feature extraction ===")
        print(f"Valid segments: {len(valid_records)}")
        print(f"Failed segments: {len(failed_records)} "
              f"({100 * len(failed_records) / max(1, len(records)):.2f}% of {len(records)})")
        print(f"X shape: {X.shape}")
        print(f"# features: {len(feature_names)}")
        print(f"Build time: {build_seconds:.1f}s "
              f"({build_seconds / max(1, len(records)):.4f}s per segment, "
              f"{build_seconds / max(1, len(tracks)):.2f}s per track)")
        if args.dataset == "compmusic_hmd" and len(tracks) < 300:
            projected = build_seconds / max(1, len(tracks)) * 300
            print(f"Projected full-300-track build: {projected / 60:.1f} min")

        if failed_records:
            reasons = Counter(str(r.get("error", "unknown"))[:70] for r in failed_records)
            print("\nFailure reasons (top 5):")
            for reason, count in reasons.most_common(5):
                print(f"  {count:5d}  {reason}")

    if X.shape[0] == 0:
        print("\nNo valid segments produced. Stop here and debug slicing / stage1.")
        return

    y = [r["raga_label"] for r in valid_records]

    n_tracks_out = len({r["track_id"] for r in valid_records})
    n_classes = len(set(y))
    per_raga = Counter(
        r["raga_label"] for r in
        {rec["track_id"]: rec for rec in valid_records}.values()
    )
    print("\nSegments per raga:", dict(Counter(y)))
    print("Tracks per raga:", dict(per_raga))

    if from_run_dir is None:
        write_manifest(run_dir, args, sys.argv, X, feature_names, n_tracks_out,
                       n_classes, build_seconds, per_raga, Counter(y))
    else:
        # An eval-only pass must not touch the manifest: it describes the
        # extraction, which this pass did not perform.
        print(f"\n  (eval-only pass: manifest.json, features.csv.gz and "
              f"tsne.png left untouched)")

    if args.skip_classification:
        print("\n--skip-classification set: feature extraction only, stopping here.")
        return

    if min(per_raga.values()) < 2:
        print("\nSkipping classification: at least one raga has a single recording, "
              "so a group-aware split cannot both train and test on it.")
        return

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

    # Features are identical across grouping schemes -- grouping only changes
    # cross-validation -- so extract once and evaluate under each scheme in
    # turn, writing a separate result file per scheme.
    feature_set = args.feature_set or f"{X.shape[1]}feat-full"
    # "62feat-passA" -> "62 (passA)" in the index's fixed-width feats column.
    tag = feature_set.split("-", 1)[1] if "-" in feature_set else None
    feats_label = f"{X.shape[1]} ({tag})" if tag and tag != "full" else str(X.shape[1])

    all_runs = {}
    for grouping in args.group_by:
        groups = [group_key_of[grouping][r["track_id"]] for r in valid_records]

        # Distinct CV groups per raga caps the usable fold count: a raga with 5
        # sessions cannot be spread over 10 folds.
        groups_per_raga = collections.Counter()
        seen: set[tuple[str, str]] = set()
        for lbl, grp in zip(y, groups):
            if (lbl, grp) not in seen:
                seen.add((lbl, grp))
                groups_per_raga[lbl] += 1

        print(f"\n\n{'#' * 70}")
        print(f"# GROUPING UNIT: {grouping.upper()}")
        print(f"{'#' * 70}")
        print(f"{len(set(groups))} distinct groups over {n_tracks_out} tracks")
        print("Groups per raga:", dict(sorted(groups_per_raga.items())))

        if args.cv == "logo":
            splitter = LeaveOneGroupOut()
            cv_desc = "LeaveOneGroupOut"
        else:
            n_splits = min(args.n_splits, min(groups_per_raga.values()))
            if n_splits < args.n_splits:
                limiting = [r for r, c in groups_per_raga.items()
                            if c == min(groups_per_raga.values())]
                print(f"  note: reducing n_splits {args.n_splits} -> {n_splits}; "
                      f"limited by {limiting}")
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=args.seed
            )
            cv_desc = f"StratifiedGroupKFold({n_splits})"
        print(f"Using {cv_desc}")

        per_fold_detail = n_tracks_out <= 60
        results = [
            evaluate_model("Logistic Regression", logreg, X, y, groups, valid_records,
                           splitter, per_fold_detail, show_confusion=n_classes <= 8),
            evaluate_model("Random Forest", rf, X, y, groups, valid_records,
                           splitter, per_fold_detail, show_confusion=n_classes <= 8),
        ]

        print(f"\n===== RESULT SUMMARY (grouped by {grouping}) =====")
        print(f"dataset={args.dataset}  tonic={mode}  cv={cv_desc}  "
              f"tracks={n_tracks_out}  ragas={n_classes}  segments={X.shape[0]}")
        print(f"{'model':<22}{'track acc':>11}{'seg acc':>10}{'seg x chance':>14}")
        for r in results:
            print(f"{r['model']:<22}{r['track_accuracy']:>11.4f}"
                  f"{r['segment_accuracy']:>10.4f}"
                  f"{r['segment_accuracy'] / r['chance']:>13.1f}x")
        print(f"chance baseline: {results[0]['chance']:.4f}")

        all_runs[grouping] = {
            "grouping": grouping,
            "cv": cv_desc,
            "n_groups": len(set(groups)),
            "groups_per_raga": dict(sorted(groups_per_raga.items())),
            "results": results,
        }
        write_results(args, grouping, all_runs[grouping], X, n_tracks_out,
                      n_classes, build_seconds, per_raga, Counter(y), run_dir,
                      feature_set=feature_set, feature_names=feature_names,
                      features_dropped=features_dropped,
                      from_run=from_run_dir is not None)
        append_to_index(run_dir, args, all_runs[grouping], n_tracks_out,
                        n_classes, X.shape[1], feats_label=feats_label)

    if len(all_runs) > 1:
        print("\n\n===== GROUPING COMPARISON =====")
        print(f"{'model':<22}" + "".join(f"{g:>16}" for g in args.group_by))
        for i, name in enumerate(["Logistic Regression", "Random Forest"]):
            row = f"{name:<22}"
            for g in args.group_by:
                r = all_runs[g]["results"][i]
                row += f"{r['segment_accuracy']:>10.4f} seg"
            print(row)
            row = f"{'  (track-level)':<22}"
            for g in args.group_by:
                r = all_runs[g]["results"][i]
                row += f"{r['track_accuracy']:>10.4f} trk"
            print(row)

    print(f"\nfeature build time: {build_seconds:.1f}s")


if __name__ == "__main__":
    main()

