# HANDOFF

Last updated: 2026-08-19. Read this before touching anything.

---

## 1. GOAL

Identify Hindustani ragas from pitch contours using interpretable, musically
meaningful features — as the MIR foundation for an eventual LLM-generated
musical commentary system.

---

## 2. FROZEN REFERENCE NUMBERS

Do not recompute, overwrite, or contradict these without explicit discussion.
All use 30 s windows / 20 s hop / 15 s minimum and 71 features.

### 2a. HEADLINE — CompMusic HMD, 30 ragas, 300 recordings

Annotated (`.tonicFine`) tonic, 20,821 segments, chance = 0.0333.

| grouping | CV | model | track acc | segment acc | ×chance |
|---|---|---|---|---|---|
| track | SGKF(10), 300 groups | LR | 0.9300 | 0.7079 | 21.2× |
| track | SGKF(10), 300 groups | RF | 0.9333 | 0.6735 | 20.2× |
| **album** | **SGKF(4), 158 groups** | **LR** | **0.8633** | **0.6648** | **19.9×** |
| album | SGKF(4), 158 groups | RF | 0.8533 | 0.6264 | 18.8× |

**Quote the album-grouped LR figures** when one number is needed — strictest
protocol run. Misclassified: 21/300 and 20/300 (track); 41/300 and 44/300 (album).
Album fold count auto-capped at 4 by Khamāj (only 4 sessions), so part of the
track→album gap is reduced training data, not the confound. Extraction: 895 s,
zero failures.

Produced by:
```bash
.venv/bin/python run_segment_lr_rf.py --dataset compmusic_hmd \
    --cv sgkf --n-splits 10 --group-by track album --annotated-tonic \
    --run-tag "_hmd-full-30raga"
```
Artifacts: `outputs/runs/2026-08-17_hmd-full-30raga_annotated/`
(`manifest.json`, `features.csv.gz`, `tsne.png`, `eval/71feat-full_by-{track,album}_lr-rf.{json,txt}`).
Headline numbers for every run are also tabulated in `outputs/INDEX.md`.

### 2b. Controlled 5-raga pilot (same ragas as Saraga, 10 recordings each)

50 tracks, 3,631 segments, chance = 0.2000. Ragas: Bihāg, Kēdār, Bhūp, Ābhōgī, Śrī.

| grouping | tonic | LR seg | RF seg | LR track |
|---|---|---|---|---|
| track SGKF(10) | estimated | 0.6673 | 0.8160 | 0.9400 |
| track SGKF(10) | annotated | 0.9391 | 0.9303 | 1.0000 |
| album SGKF(5) | estimated | 0.6502 | 0.7981 | 0.8600 |
| album SGKF(5) | annotated | **0.9193** | 0.8882 | **0.9600** |

This is the controlled proof that **data volume, not dataset identity**, drove
the improvement: same ragas as Saraga, only recordings-per-raga changed (2–3 → 10).

### 2c. Saraga baseline, 6 ragas, 13 recordings — LeaveOneGroupOut

1,809 segments, chance = 0.1667. Superseded as headline, still the reference
Saraga number.

| tonic | LR track | LR seg | RF track | RF seg |
|---|---|---|---|---|
| estimated | 0.615 | 0.375 | 0.615 | 0.384 |
| annotated | 0.615 | **0.520** | 0.538 | 0.441 |

`.venv/bin/python run_segment_lr_rf.py --annotated-tonic`

### 2d. Tonic estimator error vs ground truth (Saraga)

`tests/run_tonic_validation.py`

| level | n | within ±50¢ raw | octave-folded | pitch-class errors |
|---|---|---|---|---|
| track | 13 | 46.2% | 61.5% | 38.5% |
| **segment** | **1809** | **26.4%** | **49.8%** | **50.2%** |

Median folded error 75 cents. Swara assignment folds mod 1200, so octave errors
are largely harmless but pitch-class errors invalidate the whole vector.

### 2e. HMD dataset structure (measured)

116.1 h, 20,821 segments, 300 recordings, 30 ragas × exactly 10. Durations
1.6–71.1 min (median 21.4). Segments per raga **392 (Dēś) to 1095 (Śrī) — 2.8×
imbalance** despite balanced track counts; per-class F1 correlates **+0.38**
with segment count. Only **158 independent (artist, album) sessions**. Artist
diversity per raga 2–10; **Khamāj = 2 artists / 4 sessions**, the binding
constraint on album folds.

---

## 3. CURRENT STATE

**Datasets**
- Saraga at `/home/thesidpai/mir_projects/data` (13-track / 6-raga experiment subset).
- HMD at `data/compmusic_raga/RagaDataset/Hindustani/` — features only, 3.4 GB, no audio.
  **mirdata's `compmusic_raga` loader cannot read it** (v1.0 index = 477 Carnatic only).
  Read directly off disk; tracks joined to metadata by **MBID, never by path**
  (archive sanitises `:`, `&`, `'` → `_`; 64/300 paths don't resolve).

**Implemented and verified**
- `commentator/io/saraga.py` + `compmusic.py` — one adapter per dataset, same
  interface: `name`, `list_tracks()`, `get_pitch(id)`, `get_tonic(id)`.
  HMD prefers `.tonicFine`.
- Tonic override: `analyze_pitch_musically` → `build_stage1_schema(tonic_hz=)`
  → `build_segment_feature_dataset(get_tonic_fn=)`. Histogram still computed,
  so only normalization/swara assignment change.
- `track_id` exported in the CSV as a **string** (grouping key, never a
  feature). `segment_index` deliberately NOT exported — numeric, would be
  picked up as a feature.
- `run_segment_lr_rf.py` flags: `--dataset`, `--annotated-tonic`,
  `--cv logo|sgkf`, `--n-splits`, `--group-by track album` (multi-value: one
  extraction, N evaluations), `--ragas`, `--per-raga`, `--run-tag`,
  `--skip-classification`.
- **outputs/ layout** (restructured 2026-08-19): one directory per feature
  extraction at `outputs/runs/<run_id>/` holding `manifest.json` (command, git
  sha, params, feature names), `features.csv.gz` (named columns, gzipped) and
  `tsne.png`; every evaluation of those features goes in that run's `eval/`.
  `outputs/INDEX.md` tabulates all runs. Rule: **anything that changes the
  numbers in features.csv.gz → new run directory; anything that only changes
  how they are evaluated → a file in eval/.** `run_segment_lr_rf.py` refuses to
  write into a non-empty run directory without `--overwrite`.
- t-SNE scales to 30 classes, subsamples above 4,000 points.

**Not implemented**: estimated-tonic full 30-raga run; any classifier beyond
LR/RF; segment cap; `run_tonic_validation.py` on HMD (only a 20-track spot
check); **the tonic estimator fix** — `resolve_tonic_octave`'s octave/fifth bug
is unfixed, and annotated tonics are a workaround that won't transfer to
unannotated data.

**Git**
- Local `main` at `08c90d7`; `origin/main` at `579c441`; **2 commits unpushed**.
- Backup refs from the co-author-trailer rewrite still present and safe to delete
  once GitHub is confirmed: `pre-rewrite-backup`, `refs/original/refs/heads/main`.
- `CLAUDE.md` lives at the workspace root, **outside any git repo** — updated but
  intentionally untracked.

---

## 4. RETRACTED / DO-NOT-CITE

- **RF 0.9051 / LR 0.8035 segment accuracy.** Leaky. `classifier_compare.py`
  found no group column (the CSV omitted `track_id`) and silently fell back to
  `train_test_split(stratify=y)` — a random split over *segments*. Segments are
  30 s windows at 20 s hop, so 94.9% of test segments had an overlapping
  neighbour in training and all 13 tracks appeared on both sides. **Corrected
  under grouping: 0.2677 / 0.2402.** Fixed by exporting `track_id`.
- **"mirdata's `compmusic_raga` exposes Hindustani via `track.tradition`."**
  False. Index v1.0 = 477 recordings, all Carnatic, zero Hindustani.
- **"The tonic problem is why 0.9051 was inflated."** Wrong cause. The split was.
  The tonic issue is real but independent.
- **"73 features."** It is **71**.
- **"Khamāj will score artificially high"** (2 artists / 10 recordings). It does
  not — F1 0.598 vs 0.696 macro average. Do not drop Khamāj.
- **2026-06-16 log prose "608 valid segments at 30 s/30 s hop."** The saved
  artifact shows **910**, reproducible only at 20 s hop. Artifact wins.

---

## 5. DECIDED NEXT STEP

**Ablate the absolute-pitch / recording-property features and re-run the full
HMD set on the frozen album-grouped protocol.**

Rationale: a raga is defined independently of absolute pitch, so `tonic_hz` etc.
carry no legitimate raga information — but they do encode performer vocal range
and tanpura tuning. With only 158 sessions behind 300 recordings, they are a
plausible back-door for the artist confound the album grouping exists to
control. Either result is informative: unchanged accuracy means drop them
permanently; a drop means part of the headline was riding on performer identity.

Exact change — add these 9 names to `DROP_NAMES` in
`commentator/analysis/raga_features.py` (currently 11 entries, verified present
in the live 71-feature vector):

```
tonic_hz, log_tonic_hz, target_hz, n_tonic_candidates,
n_voiced_frames, log_n_voiced_frames, n_confident_frames,
log_n_confident_frames, confident_ratio
```

**No re-extraction is needed.** `apply_feature_subset` is pure column masking,
so dropping these columns from the saved
`outputs/runs/2026-08-17_hmd-full-30raga_annotated/features.csv.gz` gives
numerically identical results to re-extracting (imputation and scaling are
per-column). All three ablation passes are column subsets of that one table —
they belong in its `eval/` as `65feat-passA_*`, `59feat-passB_*`,
`52feat-passC_*`, not in new run directories. Only *adding* features
(transitions, nyas) requires re-extraction.

Expect 62 features after Pass A. Compare against §2a.

---

## 6. QUEUED AFTER THAT

1. Wire a CSV-based classifier sweep (extend `tests/classifier_compare.py`) so classifier variants cost seconds, not 15 min re-extraction.
2. Sweep classifiers on the frozen protocol — `HistGradientBoostingClassifier` first (most likely win), then LinearSVC, RBF-SVM, LDA, MLP.
3. Permutation importance on the surviving feature set to find dead weight.
4. Add swara **transition/bigram** features (144 dims) — targets the measured Kēdār/Bihāg confusion, which distribution-only features cannot see.
5. Add **nyas** features from the unused `.flatSegNyas` files (300 present on disk).
6. Segment-cap experiment (cap per recording at ~64, the median) against the 2.8× imbalance.
7. Window-length sweep: 30 / 60 / 120 s.
8. Estimated-tonic variant of the full 30-raga run, to confirm the tonic effect at 30 classes.
9. `run_tonic_validation.py` across all 300 HMD tracks.
10. Confusion-structure / thaat clustering analysis for the writeup.
11. Fix `resolve_tonic_octave` — the one unaddressed root cause.

---

## 7. OPEN QUESTIONS

- **§5 was recommended, not explicitly confirmed** — the handoff was requested
  before an answer. Confirm before running.
- **Multiple-comparisons risk.** ~30 planned variants on one CV; picking the best
  will overstate. Defence undecided: log every variant, or decide on the 5-raga
  pilot and confirm only finalists on the full 300 (recommended), or hold out
  recordings entirely.
- **Segment cap is an experimental variable, not cleanup** — report alongside
  the uncapped result, not instead of it. Not yet agreed.
- **Annotations assumed correct.** Neither Saraga's `ctonic` nor HMD's
  `.tonicFine` independently verified; all tonic error rates rest on them.
- **Track-level vs segment-level as headline.** Track-level is fairer given the
  2.8× segment imbalance; segment-level is what MIR papers usually report.
  Currently reporting both.
