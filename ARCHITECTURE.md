# Architecture

How the code is laid out and what happens when the pipeline runs.

**This file deliberately contains no results.** `HANDOFF.md` is the single
authority on frozen numbers, current state and next steps; the dated history is
in `docs/experiments/2026-06-raga-baseline-log.md`. This file was `status.txt`
until 2026-08-19, where it duplicated HANDOFF's numbers and drifted out of sync
with them — two "read this first" documents that can disagree is one too many.

---

## Repository tree

```
.
├── commentator/                 the package: library code only, no drivers
│   ├── analysis/
│   │   ├── pipeline_one.py      Stage-1 chain (analyze_pitch_musically)
│   │   ├── stage1_schema.py     build_stage1_schema -- Stage-1 entry point
│   │   ├── tonic_estimator.py   tonic estimation + normalization
│   │   ├── swara_analyzer.py    swara assignment + summary
│   │   ├── raga_features.py     Stage-2 feature vector + apply_feature_subset
│   │   ├── segment_dataset.py   windowing + build_segment_feature_dataset
│   │   └── raga_baseline_knn.py older KNN baseline (superseded)
│   ├── core/pitch_contour.py    PitchContour -- the interchange type
│   ├── io/
│   │   ├── saraga.py            SaragaHindustani adapter
│   │   └── compmusic.py         CompMusicHindustani (HMD) adapter
│   ├── tests/test_pitch_contour.py   the repo's only pytest test
│   └── utils/audio_utils.py
├── scripts/                     drivers that consume the package (not tests)
│   ├── README.md                what each script is for
│   └── legacy/                  superseded scripts, kept for reproducibility
├── docs/
│   ├── project-report.md        outline/skeleton, not filled in
│   └── experiments/
│       ├── 2026-06-raga-baseline-log.md   the dated history
│       └── tsne/                archived historical t-SNE snapshots
├── outputs/
│   ├── INDEX.md                 one row per evaluation, all runs
│   ├── runs/<run_id>/           one directory per FEATURE EXTRACTION
│   │   ├── manifest.json        command, git sha, params, feature names
│   │   ├── features.csv.gz      named columns, includes track_id
│   │   ├── tsne.png
│   │   └── eval/                every evaluation of those features
│   ├── commentary/              LLM commentary results
│   ├── inspect_features/        exploratory track-level inspection
│   └── legacy/                  superseded artifacts (see its README)
├── ARCHITECTURE.md              this file
├── HANDOFF.md                   numbers, state, next steps -- read first
└── run_segment_lr_rf.py         the main entry point
```

`.git`, `.venv`, `__pycache__` and `.pytest_cache` are omitted; the latter two
are gitignored, as is `.env`.

**The layout rule for `outputs/`:** anything that changes the numbers inside
`features.csv.gz` (dataset, track subset, tonic mode, window, added features)
gets a **new run directory**; anything that only changes how those numbers are
*evaluated* (grouping, classifier, CV, feature subsets) is a **file in
`eval/`**. Filenames inside a run are fixed and boring — the directory name and
manifest carry the identity.

---

## What happens when `run_segment_lr_rf.py` runs

1. The script selects a dataset adapter — `commentator/io/saraga.py` or
   `commentator/io/compmusic.py`. Both expose `name`, `list_tracks()`,
   `get_pitch(track_id)`, `get_tonic(track_id)`, so nothing downstream changes
   between them.
2. For each chosen recording it fetches a `PitchContour`
   (`commentator/core/pitch_contour.py`) — times, frequencies, voiced mask,
   confidence. This is the interchange type through the whole pipeline.
3. `build_segment_feature_dataset(...)` in
   `commentator/analysis/segment_dataset.py` infers duration from
   `pitch_obj.times` and cuts **30 s windows at a 20 s hop**, minimum 15 s.
4. Each window becomes a smaller `PitchContour` via `slice_pitch_contour(...)`.
5. That slice goes to `build_stage1_schema(...)`
   (`commentator/analysis/stage1_schema.py`), optionally with an annotated
   `tonic_hz` supplied by the adapter instead of an estimate.
6. `build_stage1_schema` calls `analyze_pitch_musically(...)` in
   `commentator/analysis/pipeline_one.py`, which runs the Stage-1 chain:
   tonic estimation and normalization (`tonic_estimator.py`), then swara
   assignment and summary (`swara_analyzer.py`).
7. `extract_raga_features_from_stage1(...)`
   (`commentator/analysis/raga_features.py`) turns that Stage-1 dict into a
   numeric vector: **82 dimensions raw, 71 after `apply_feature_subset`** drops
   the 11 names in `DROP_NAMES`.
8. `build_segment_feature_dataset` collects valid segments into `X`,
   `feature_names` and `records`, drops failures, and writes
   `features.csv.gz` + `tsne.png` into `run_dir`. `track_id` is exported **as a
   string** — it is a grouping key, never a feature, and omitting it once caused
   a leak. `segment_index` is deliberately not exported.
9. Back in `run_segment_lr_rf.py`, `y` comes from `raga_label` and groups from
   either `track_id` or an artist+album **session** key.
10. **Evaluation is always group-aware**: `LeaveOneGroupOut` for Saraga's 13
    recordings, `StratifiedGroupKFold` for HMD. Segments are overlapping windows
    of one performance, so a split that scatters a recording across train and
    test leaks near-duplicate rows.
11. Two pipelines are fitted per grouping — `SimpleImputer → StandardScaler →
    LogisticRegression`, and `SimpleImputer → RandomForestClassifier`.
12. Segment predictions are **majority-voted into one track-level prediction**.
13. Results are written to `runs/<run_id>/eval/`, one JSON and one readable
    report per grouping, each recording which feature columns it used and
    dropped. A row per evaluation is appended to `outputs/INDEX.md`.

### Evaluating a saved run without re-extracting

`--from-run <run_id>` loads that run's `features.csv.gz`, masks columns with
`--drop-features`, and reuses the identical evaluation path from step 9 onward.
Dataset, tonic mode and window come from the run's `manifest.json`, never from
retyped flags. It writes only into `eval/`.

Because `apply_feature_subset` is pure column masking and imputation and scaling
are fitted per column, evaluating a subset of a saved table is numerically
identical to re-extracting without those features. **Run the control first** —
the same command with no `--drop-features` must reproduce the run's saved
numbers exactly, since folds depend only on labels, groups and seed.

### The classifier's feature set is an evaluation choice

Extraction produces all 71 columns. The classifier currently masks down to 36 at
evaluation time (HANDOFF §2g). Those 35 names must **not** move into
`DROP_NAMES`: the contour/stability features among them are dead weight for
classification but describe movement — meend, stability, transition density —
which the commentary system is likely to need.

---

## Files involved in the main flow

```
run_segment_lr_rf.py
commentator/io/{saraga,compmusic}.py
commentator/core/pitch_contour.py
commentator/analysis/segment_dataset.py
commentator/analysis/stage1_schema.py
commentator/analysis/pipeline_one.py
commentator/analysis/tonic_estimator.py
commentator/analysis/swara_analyzer.py
commentator/analysis/raga_features.py
```

## Datasets

Two, each behind an adapter exposing the same interface, so pipeline code can be
pointed at either unchanged.

- **SaragaHindustani** (`commentator/io/saraga.py`) — 108 tracks, but only 6
  ragas have ≥2 usable recordings; experiments use a 13-track subset.
  `DATA_HOME` is `/home/thesidpai/mir_projects/data`.
- **CompMusicHindustani** (`commentator/io/compmusic.py`) — the HMD subset: 300
  recordings, 30 ragas, exactly 10 each. Read directly off
  `data/compmusic_raga/RagaDataset/Hindustani/`, because mirdata's
  `compmusic_raga` index v1.0 covers only the 477 Carnatic recordings. Tracks
  join to metadata **by MBID, never by path** — the archive sanitises `:`, `&`
  and `'` to `_`, so 64 of 300 recorded paths do not resolve. Prefers
  `.tonicFine` for the annotated tonic. Audio is never touched.

## Conventions

- New pipeline stages go in the relevant `commentator/analysis/*.py` module and
  are wired through `commentator/analysis/__init__.py`'s `__all__`, following
  the Stage-1 / Stage-2 split rather than reaching into `pipeline_one.py`.
- `PitchContour` is the interchange type between `io`, `core` and `analysis`.
  New loaders and synthetic-data helpers should produce or consume it, not raw
  arrays.
- `commentator/` holds library code. Anything runnable is a driver and belongs
  in `scripts/`, or at the root if it is the main program.
- Track IDs differ per dataset: Saraga uses mirdata's `<index>_<Raga_Name>`
  convention; HMD track IDs are MusicBrainz IDs with transliterated raga labels
  carrying diacritics (`"Bihāg"`, `"Bhūp"` = Bhoopali).
- Generated artifacts must never collide across runs. One run directory per
  extraction, fixed filenames inside, identity in the directory name and
  manifest.
