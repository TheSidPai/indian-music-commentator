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

### 2f. Per-feature diagnostics (measured on the 20,821 x 71 HMD table)

Basis for the §5 ablations. "ratio" = variance explained by track identity /
variance explained by raga; high means the feature describes the recording
rather than the music.

| family | n | ratio | note |
|---|---|---|---|
| `swara_prop_*` | 12 | **1.2** | genuine raga signal — keep |
| `log_swara_count_*` | 12 | **1.3** | genuine raga signal — keep (see §5) |
| `trigram_prop_*` | 6 | 6.9 | MI 0.000–0.008, near-dead |
| `range_span_cents` | 1 | 7.5 | performer tessitura |
| contour/stability | 16 | **8.5** | performance style, not raga |

Individual features worth knowing:

- **`hist_ref_hz` is constant at 55.0** across all 20,821 rows. Zero variance, MI 0.0000 — a hardcoded cents reference leaking into the vector. **The live feature count is therefore 70, not 71**; dropping it is provably a no-op.
- **`tonic_hz` is constant within 100% of recordings**, 234 distinct values across 300 tracks — a near-unique session fingerprint. It has the **highest MI with raga of all 71 features (3.04)**, five times the best genuine feature (`hist_peak_1_cents`, 0.68). That is memorisation capacity, not musical signal, and it is exactly the back-door §5 Pass A closes.
- Near-duplicate pairs (|r| > 0.97): `tonic_hz`/`log_tonic_hz` 0.998; `bigram_prop_Ni_Sa`/`Sa_Ni` 0.997; `mean_pitch_step_size_cents` vs `mean_positive/negative_pitch_diff` 0.985/0.988; `stable_frame_ratio`/`transition_frame_ratio` 0.985.
- The 12 `swara_prop_*` sum to exactly 1.0 — compositional, so one is determined by the other eleven.

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
- Snapshot at time of writing: `main` at `60577d5`, `origin/main` at `626f4db`,
  **1 commit unpushed**. Verify with `git status -sb` rather than trusting this line.
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

## 5. DECIDED NEXT STEP — feature ablation, three passes

All three agreed. Evidence for every drop is in §2f. **Filenames use the actual
surviving count** (an earlier draft of this section said 65/59/52; that was a
stale carry-over and is wrong — ignore it).

### Harness

Add `--from-run <run_id> --drop-features <names...> --feature-set <name>` to
**`run_segment_lr_rf.py`**, loading `features.csv.gz` and skipping extraction.
`apply_feature_subset` is pure column masking, so dropping columns from the
saved table is numerically identical to re-extracting (imputation and scaling
are per-column) — no re-extraction for any pass.

**Do not use `tests/classifier_compare.py`** for these. It splits with a single
`GroupShuffleSplit`, the same family of shortcut that produced the retracted
0.9051. The frozen protocol — SGKF, album group map, segment→track majority
vote, seed 42 — exists only in `run_segment_lr_rf.py`, and reusing that exact
code path is what makes A/B/C comparable to §2a.

### Control pass first — this gates everything

Re-evaluate the untouched 71 columns from the saved CSV and require an **exact**
reproduction of §2a (album LR 0.8633 / 0.6648; track LR 0.9300 / 0.7079). Folds
depend only on labels, groups and seed, not on features, so with row order
preserved this must match to the digit. **If it does not match exactly, stop** —
no ablation delta is attributable to columns until it does.

### The passes

Run each with `--group-by track album` (one load, two evaluations). The album
number is the report number; the track−album gap is itself a diagnostic.

| pass | drops | surviving |
|---|---|---|
| control | — | 71 |
| **A** — confounds | `tonic_hz`, `log_tonic_hz`, `target_hz`, `n_tonic_candidates`, `n_voiced_frames`, `log_n_voiced_frames`, `n_confident_frames`, `log_n_confident_frames`, `confident_ratio` | **62** |
| **B** — dead weight | + `hist_ref_hz`, all 6 `trigram_prop_*` | **55** |
| **C** — performance style | + the 16 contour/stability features (`mean_abs_pitch_diff_cents`, `std_pitch_diff_cents`, `mean_positive_pitch_diff_cents`, `mean_negative_pitch_diff_cents`, `frac_rising_frames`, `frac_falling_frames`, `frac_flat_frames`, `mean_pitch_step_size_cents`, `n_stable_regions`, `mean_stable_region_len`, `max_stable_region_len`, `stable_frame_ratio`, `n_transition_regions`, `mean_transition_region_len`, `max_transition_region_len`, `transition_frame_ratio`), plus `range_span_cents`, `unassigned_frames`, `log_unassigned_frames` | **36** |

C leaves a clean, defensible set: 12 `swara_prop_*` + 12 `log_swara_count_*` +
4 histogram-shape + 8 `bigram_prop_*`. All tonic-relative and raga-theoretic.

**Do NOT drop `log_swara_count_*`.** A plausible-sounding argument says counts =
proportions × voiced density, so they smuggle back a recording property. The
data says otherwise: track/raga variance ratio **1.3**, essentially identical to
`swara_prop_*` at 1.2, and `log_swara_count_re` has the second-highest mutual
information of any feature (0.46). They are raga signal, not confound.

### Expectations

- **A**: album LR should move little (the 24 swara dims carry the signal). Two directional checks: **RF should lose more than LR** (it splits greedily on high-cardinality continuous columns, and `tonic_hz` is a near-unique session id); and **track-grouped should lose more than album-grouped**, narrowing the 0.93/0.86 gap, since under track grouping a held-out track's album siblings sit in training with nearly the same tonic. Accuracy going slightly *up* is plausible — fewer confounded dims can help LR's regularisation.
- **B**: near no-op by construction (`hist_ref_hz` is provably inert). Expect ≤0.01. It is the attribution step, not a finding.
- **C**: most likely to cost accuracy — 19 live dims. Guess 0.02–0.05 off album LR. If flat, adopt the 36-dim set permanently.

### Statistical caveat, to be carried into the writeup

At n=300 tracks and p≈0.86 the binomial SE is ≈0.020, so **any track-accuracy
difference under ~4 points is inside 2 SE and not distinguishable from noise**.
Three passes × two groupings × two models = 12 numbers on one CV. Log all 12;
read only large, directionally consistent moves as real. This is §7's
multiple-comparisons concern in concrete form.

### Deliverables

`62feat-passA_by-{track,album}_lr-rf.{json,txt}`, likewise `55feat-passB` and
`36feat-passC`, plus `71feat-control_*`, all into
`outputs/runs/2026-08-17_hmd-full-30raga_annotated/eval/`. INDEX.md rows
appended. §2a left untouched.

## 6. QUEUED AFTER THAT

1. ~~Wire a CSV-based classifier sweep~~ — partly done: `classifier_compare.py --run <run_id>` now reads a run's `features.csv.gz` and writes into its `eval/`. Still only LR+RF; extend it to take a list of classifiers and a `--drop-features` argument (the latter is what makes the §5 ablations one-liners).
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

- **§5 is agreed** (2026-08-19): all three ablation passes approved. Pass A is
  the immediate action; B and C follow as separate passes so any change is
  attributable to a specific removal.
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
