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
All use 30 s windows / 20 s hop / 15 s minimum. Feature count is **71 except in
§2g**, which is the 36-feature ablation result and the current headline.

### 2a. CompMusic HMD, 30 ragas, 300 recordings — 71 features

> **Superseded as headline by §2g (2026-08-19).** These figures are **not
> retracted** — they were reproduced exactly by the ablation control pass and
> remain the correct numbers **for the 71-feature set**, which is the right
> comparison point for it. Quote §2g for the current headline.

Annotated (`.tonicFine`) tonic, 20,821 segments, chance = 0.0333.

| grouping | CV | model | track acc | segment acc | ×chance |
|---|---|---|---|---|---|
| track | SGKF(10), 300 groups | LR | 0.9300 | 0.7079 | 21.2× |
| track | SGKF(10), 300 groups | RF | 0.9333 | 0.6735 | 20.2× |
| **album** | **SGKF(4), 158 groups** | **LR** | **0.8633** | **0.6648** | **19.9×** |
| album | SGKF(4), 158 groups | RF | 0.8533 | 0.6264 | 18.8× |

Misclassified: 21/300 and 20/300 (track); 41/300 and 44/300 (album).
Album fold count auto-capped at 4 by Khamāj (only 4 sessions). Extraction: 895 s,
zero failures.

**Correction (2026-08-19):** this section previously read that part of the
track→album gap here is reduced training data from the 4-fold cap "not the
confound". Measured by the §2g ablation, most of it *was* the confound — the LR
gap is 0.0667 at 71 features and 0.0267 at 36. The fold cap explains at most the
smaller residual.

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

`scripts/run_tonic_validation.py`

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

### 2g. HEADLINE — HMD 30 ragas / 300 recordings, 36 features

The §5 ablation, run 2026-08-19 on the same 20,821 × 71 table as §2a (no
re-extraction — pure column masking). Annotated tonic, chance = 0.0333.

**Album-grouped, SGKF(4), 158 groups — the frozen protocol, quote these:**

| pass | feats | LR track | LR seg | RF track | RF seg |
|---|---|---|---|---|---|
| control | 71 | 0.8633 | 0.6648 | 0.8533 | 0.6264 |
| A | 62 | 0.9267 | 0.7006 | 0.9267 | 0.6669 |
| B | 55 | 0.9233 | 0.7000 | 0.9367 | 0.6657 |
| **C** | **36** | **0.9300** | **0.7179** | **0.9300** | **0.6631** |

**Track-grouped, SGKF(10), 300 groups:**

| pass | feats | LR track | LR seg | RF track | RF seg |
|---|---|---|---|---|---|
| control | 71 | 0.9300 | 0.7079 | 0.9333 | 0.6735 |
| A | 62 | 0.9500 | 0.7224 | 0.9233 | 0.6718 |
| B | 55 | 0.9500 | 0.7236 | 0.9300 | 0.6704 |
| **C** | **36** | **0.9567** | **0.7288** | 0.9333 | 0.6664 |

**When one number is needed: album-grouped LR, 36 features, 0.9300 track
accuracy / 0.7179 segment (21.5× chance), 21/300 recordings misclassified.**
Macro F1 0.7009, up from 0.6478 at 71 features.

**The 36 surviving features** — 12 `swara_prop_*`, 12 `log_swara_count_*`,
4 histogram-shape (`hist_peak_1_cents`, `hist_peak_1_height`, `hist_entropy`,
`hist_concentration`), 8 `bigram_prop_*`. Spelled out here because the eval
artifacts record only the feature *count*, not the names (see §3).

**Significance — paired McNemar, not a binomial band.** Every pass predicts the
same 300 recordings under the same folds (folds depend only on labels, groups
and seed), so the comparison is paired. All three passes beat the control
album-grouped: LR fixes 23/21/26 recordings and breaks 4/3/6 (p = 0.0003,
0.0003, 0.0005); RF fixes 30/31/30 and breaks 8/6/7 (p = 0.0005, <0.0001,
0.0002). Survives Bonferroni over all 24 pairwise tests run.
**A, B and C are mutually indistinguishable** (p = 0.25–1.00), and *nothing*
changes significantly under track grouping. C is chosen on parsimony, not
because 0.9300 > 0.9267 — that is one recording.

**Why it went up.** The Pass-A features were an active liability, not ballast.
Textbook confound signature: harmless under track grouping (a held-out
recording's album siblings share its tonic, so the cue still works), harmful
under album grouping (session unseen, so the cue misleads). `tonic_hz` alone had
the highest MI with raga of all 71 features — memorisation capacity being spent
on the wrong thing.

Produced by (drop lists in §5):
```bash
.venv/bin/python run_segment_lr_rf.py --from-run 2026-08-17_hmd-full-30raga_annotated \
    --cv sgkf --n-splits 10 --group-by track album \
    --feature-set 36feat-passC --drop-features <35 names>
```
Artifacts: `outputs/runs/2026-08-17_hmd-full-30raga_annotated/eval/{71feat-control,62feat-passA,55feat-passB,36feat-passC}_by-{track,album}_lr-rf.{json,txt}`.

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
  `--skip-classification`, and (2026-08-19) `--from-run <run_id>`,
  `--drop-features <names...>`, `--feature-set <name>`.
- **Eval-only mode** (`--from-run`): loads a saved run's `features.csv.gz`,
  masks columns, and reuses the identical evaluation path — same SGKF, album
  group map, majority vote, seed. Dataset/tonic/window are read from that run's
  `manifest.json`, not retyped on the command line. It never rewrites
  `manifest.json`, `features.csv.gz` or `tsne.png`, and refuses to overwrite an
  existing `eval/` result without `--overwrite`. **Verified**: re-evaluating all
  71 columns reproduces §2a to the digit, misclassification counts included.
  That control is a mandatory gate before trusting any ablation delta.
- **The classifier's feature set is an evaluation choice, not an extraction
  choice.** Extraction stays at 71 columns; the classifier masks to 36 via
  `--drop-features`. Do **not** move the §2g drops into `DROP_NAMES` — the 16
  contour/stability features are dead weight for classification but describe
  movement (meend, stability, transition density), which the commentary system
  is likely to need.
- **outputs/ layout** (restructured 2026-08-19): one directory per feature
  extraction at `outputs/runs/<run_id>/` holding `manifest.json` (command, git
  sha, params, feature names), `features.csv.gz` (named columns, gzipped) and
  `tsne.png`; every evaluation of those features goes in that run's `eval/`.
  `outputs/INDEX.md` tabulates all runs. Rule: **anything that changes the
  numbers in features.csv.gz → new run directory; anything that only changes
  how they are evaluated → a file in eval/.** `run_segment_lr_rf.py` refuses to
  write into a non-empty run directory without `--overwrite`.
- t-SNE scales to 30 classes, subsamples above 4,000 points.
- **Repo layout** (decluttered 2026-08-19): `tests/` → **`scripts/`**, because it
  contained zero tests — the only pytest test is
  `commentator/tests/test_pitch_contour.py`. `commentator/` is library code;
  anything runnable is a driver in `scripts/`, or at the root if it is the main
  program. `scripts/README.md` describes each one. Superseded code lives in
  `scripts/legacy/`, superseded artifacts in `outputs/legacy/` — **never cite
  either**. `status.txt` became **`ARCHITECTURE.md`**, which holds the tree and
  the pipeline walkthrough and deliberately **no results**, so it can no longer
  contradict this file.

**Eval provenance — fixed 2026-08-19.** The eval artifacts used to record
`n_features` (the count) but never *which* features, and **a count cannot be
inverted back to a set**, so a feature-subset evaluation was not reproducible
from its own files. Each eval JSON now carries `feature_set`, `features_used`,
`features_dropped`, `n_features_in_table`, `evaluated_from`, `eval_command` and
`git_commit`; each TXT leads with a one-line feature-set summary and lists both
name sets. `extraction_build_seconds` is `null` on an eval-only pass rather than
`0.0`, which previously read as "extraction was instant" instead of "not
measured". The four 2026-08-19 passes were re-run under `--overwrite` to
backfill; all 16 figures were verified byte-identical to the first run.

**Not implemented**: estimated-tonic full 30-raga run; any classifier beyond
LR/RF; segment cap; `run_tonic_validation.py` on HMD (only a 20-track spot
check); **the tonic estimator fix** —
`resolve_tonic_octave`'s octave/fifth bug is unfixed, and annotated tonics are a
workaround that won't transfer to unannotated data.

**Git**
- Snapshot at time of writing: the 2026-08-19 ablation work is committed at
  `e07a8f7`, with the `tests/` → `scripts/` declutter on top. Both sit ahead of
  `origin/main`. Verify with `git status -sb` rather than trusting this line.
- **`.git` is 146 MB, ~95 MB of it embedded base64 audio** in two notebook
  cells (`test.ipynb`, `commentator.ipynb`) from
  `IPython.display.Audio(track.audio_path)` outputs — despite the pipeline
  never touching audio. Reclaiming it needs a history rewrite and force-push;
  not done, and a decision rather than a cleanup. `nbstripout` or a pre-commit
  hook would stop further growth without rewriting anything.
- The co-author-trailer rewrite's backup refs (`pre-rewrite-backup`,
  `refs/original/refs/heads/main`) are **gone** — cleaned up once the rewritten
  history was confirmed on GitHub. Nothing left to delete.
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

## 5. DONE (2026-08-19) — feature ablation, three passes

**Completed. Results are §2g; narrative is the 2026-08-19 entry in
`docs/experiments/2026-06-raga-baseline-log.md`.** The drop lists below are now
also recorded in each eval JSON/TXT (`features_dropped`), so this section is no
longer the single point of failure it was — but keep it as the human-readable
statement of *why* each group was dropped, which the artifacts do not carry.

**Outcome in one line:** the album-grouped headline rose 0.8633 → 0.9300 track
accuracy while features fell 71 → 36. Pass A did essentially all the work;
B and C are statistically free.

**The prediction was wrong in both direction and size.** A was expected to
"move little", RF to lose more than LR, and track-grouped to lose more than
album-grouped. In fact nothing lost significantly anywhere, RF *gained* more
than LR, and the album-grouped improvement was 6.3 points. The confound
features were an active liability, not ballast — see §2g.

The harness (`--from-run`) and the control gate are described in §3; the control
reproduced §2a exactly, which is what licensed everything below.

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

### The statistical test — corrected

The plan proposed a binomial SE of ≈0.020 at n=300 and a "~4 points is noise"
band. **That test was wrong for this design.** It assumes two independent
samples, but every pass predicts the same 300 recordings under the same folds,
so the comparison is *paired*. The correct test is **McNemar on the discordant
recordings**, and it is considerably more powerful — see §2g. Use it for any
future comparison across feature sets, classifiers or hyperparameters on a
fixed CV; reserve the binomial band for genuinely independent samples.

### Deliverables — all written

`71feat-control_*`, `62feat-passA_*`, `55feat-passB_*`, `36feat-passC_*`, each
`_by-{track,album}_lr-rf.{json,txt}`, in
`outputs/runs/2026-08-17_hmd-full-30raga_annotated/eval/`. 16 INDEX.md rows
appended with the pass name in the feats column. The extraction itself
(`manifest.json`, `features.csv.gz`, `tsne.png`) is byte-unchanged.

## 6. QUEUED AFTER THAT

0. ~~Eval-provenance fix~~ — **done 2026-08-19**, see §3.
1. ~~Wire a CSV-based classifier sweep~~ — **done differently.** `--from-run` +
   `--drop-features` landed on `run_segment_lr_rf.py`, not
   `classifier_compare.py`, so the sweep reuses the frozen protocol rather than
   `classifier_compare.py`'s single `GroupShuffleSplit`. Extending
   `classifier_compare.py` is **no longer the plan**; extend
   `run_segment_lr_rf.py` to take a list of classifiers instead.
2. Sweep classifiers on the frozen protocol, **on the 36-feature set** —
   `HistGradientBoostingClassifier` first (most likely win), then LinearSVC,
   RBF-SVM, LDA, MLP. Half the features, so cheaper than planned.
3. Permutation importance on the surviving 36 to find any remaining dead weight.
4. Add swara **transition/bigram** features (144 dims) — targets the Kēdār/Bihāg
   confusion. Better motivated now: Kēdār improved to F1 0.780 under §2g, but
   `bigram_prop_*` is still only 8 hand-picked pairs. Needs re-extraction.
5. Add **nyas** features from the unused `.flatSegNyas` files (300 present on disk).
6. Segment-cap experiment (cap per recording at ~64, the median) against the 2.8× imbalance.
7. Window-length sweep: 30 / 60 / 120 s.
8. Estimated-tonic variant of the full 30-raga run — now **more interesting**:
   with `tonic_hz` dropped from the classifier, estimator errors may bite
   differently than they did at 71 features.
9. `run_tonic_validation.py` across all 300 HMD tracks.
10. Confusion-structure / thaat clustering analysis for the writeup.
11. Fix `resolve_tonic_octave` — the one unaddressed root cause.

---

## 7. OPEN QUESTIONS

- ~~**§5 is agreed**~~ — **done 2026-08-19**, results in §2g.
- **Multiple-comparisons risk.** ~30 planned variants on one CV; picking the best
  will overstate. Defence still undecided: log every variant, or decide on the
  5-raga pilot and confirm only finalists on the full 300 (recommended), or hold
  out recordings entirely.
  Partial progress: the §5 ablation adopted **paired McNemar plus Bonferroni over
  every test run** (24), which is a defence against *reading noise as signal* but
  **not** against selecting a winner from many variants. The §5 passes also came
  out mutually indistinguishable, so C was chosen on parsimony rather than on its
  score — the right move, and the one to repeat. A held-out set of recordings is
  still the only real answer once the classifier sweep starts.
- **Should the 36-feature set be re-derived per experiment?** It was selected on
  this CV, on this dataset. Treating it as fixed for all future work imports that
  selection into every later result. Cheap mitigation: re-run the control and
  Pass C whenever the dataset or window changes.
- **Segment cap is an experimental variable, not cleanup** — report alongside
  the uncapped result, not instead of it. Not yet agreed.
- **Annotations assumed correct.** Neither Saraga's `ctonic` nor HMD's
  `.tonicFine` independently verified; all tonic error rates rest on them.
- **Track-level vs segment-level as headline.** Track-level is fairer given the
  2.8× segment imbalance; segment-level is what MIR papers usually report.
  Currently reporting both.
