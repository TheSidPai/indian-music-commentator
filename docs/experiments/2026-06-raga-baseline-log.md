# Raga Baseline Experiment Log (June 2026)

Goal: Explore simple, interpretable baselines for Hindustani raga recognition on the Saraga Hindustani dataset using tonic-normalised pitch and swara features.

## Metadata

- Dataset: Saraga Hindustani (HMD)
- Codebase: indian-music-commentator
- Feature set: Stage‑1 + Stage‑2 (52‑dim feature vector)
- Models: KNN (track‑level), KNN (segment‑level)
- Ragas (current subset): Bihag, Kedar, Bhoopali

---


## Ragas with two or more tracks (from Saraga Hindustani list)

Command used to inspect track IDs:

```python
import mirdata

saraga = mirdata.initialize("saraga_hindustani", data_home=DATA_HOME)
idx = saraga.track_ids
print(idx)
```

Ragas with ≥2 tracks are:

- Raag_Shree  
  - 0_Raag_Shree  
  - 26_Raag_Shree  
  - 37_Raag_Shree  
  - 100_Raga_Shree_-_Khayal  (title variant but same raga)  
  - 107_Raag_Shree  

- Raag_Dhani  
  - 3_Raag_Dhani  
  - 19_Raag_Dhani  

- Raag_Bhimpalasi  
  - 4_Raag_Bhimpalasi  
  - 39_Raag_Bhimpalasi  
  - 46_Raag_Bhimpalasi  

- Raag_Bageshree  
  - 7_Raag_Bageshree  

- Raag_Kedar  
  - 8_Raag_Kedar  
  - 84_Raag_Kedar  

- Raag_Jog  
  - 9_Raag_Jog  
  - 40_Raag_Jog  
  - 58_Raag_Jog  

- Raag_Lalit  
  - 10_Raag_Lalit  
  - 33_Raag_Lalit  
  - 104_Raga_Lalit_-_Khayal  

- Raag_Marwa  
  - 11_Raag_Marwa  
  - 25_Raag_Marwa  
  - 74_Raag_Marwa  

- Raag_Yaman  
  - 12_Raag_Yaman  
  - 31_Raag_Yaman  

- Raag_Mian_Malhar / Raag_Miyan_Malhar  
  - 17_Raag_Miyan_Malhar  
  - 23_Raag_Mian_Malhar  
  - 28_Raag_Mian_Malhar  
  - 32_Raag_Gaud_Malhar (related, but different label)  

- Raag_Abhogi  
  - 20_Raag_Abhogi  
  - 44_Raag_Abhogi  

- Raag_Bahar  
  - 30_Raag_Bahar  
  - 48_Raag_Bahar  

- Raag_Gaud_Malhar  
  - 32_Raag_Gaud_Malhar  
  - 79_Raag_Gaud_Malhar  

- Raag_Todi  
  - 53_Raag_Todi  
  - 77_Raag_Todi  
  - 64_Todi (title variant)  

- Raag_Malkauns  
  - 56_Raag_Malkauns  
  - 62_Malkauns  

- Raag_Hameer  
  - 57_Raag_Hameer  
  - 75_Raag_Hameer  

- Raag_Bihag  
  - 27_Raag_Bihag  
  - 81_Raag_Bihag  

- Raag_Bhoopali  
  - 83_Raag_Bhoopali  
  - 105_Raag_Bhoopali  


## 2026‑06‑15 – Stage‑1 and Stage‑2 features stabilised

**Goal**

- Define a consistent Stage‑1 schema and Stage‑2 feature vector for one track.

**Changes**

- Implemented `build_stage1_schema(...)` emitting:
  - tonic section (tonic_hz, tonic_pc_cents, n_tonic_candidates, …)
  - swara summary (swara_prop_* for Sa, re, Re, ga, …)
  - pitch histogram (folded, peaks, entropy, concentration)
- Implemented `extract_raga_features_from_stage1(...)`:
  - 52 features per track including tonic, swara proportions, histogram peaks, entropy.

**Evidence**

- Sanity test on `27_Raag_Bihag`: printed feature vector and commentary match expected tonic and swara usage.

**Notes / assumptions**

- Features are tonic‑normalised; absolute pitch is considered non‑informative for raga ID at this stage.

---

## 2026‑06‑15 – Track‑level KNN baseline (3 ragas)

**Goal**

- First end‑to‑end baseline on 3 ragas.

**Setup**

- Ragas: Bihag, Yaman, Kedar.
- Tracks:
  - Raag_Bihag: 27, 81
  - Raag_Yaman: 12 (~5 min), 31 (~18 min)
  - Raag_Kedar: 8, 84
- Model: KNN (k = 3, Euclidean), StandardScaler, leave‑one‑track‑out CV.

**Result**

- Track‑level accuracy: 2/6.
- Qualitative:
  - Yaman generally separates better.
  - Bihag vs Kedar often confused.

**Decision**

- Keep this as Baseline v1; focus next on diagnosing features and within‑raga variation.

---

## 2026‑06‑16 – Segment‑level baseline v1 (Bihag/Yaman/Kedar)

**Goal**

- See if segment‑level features plus majority vote improve separation.

**Setup**

- Segment_length = 30 s, hop = 30 s, min_duration = 15 s.
- Duration from `pitch_obj.times`.
- 443 valid segments, 52‑dim features.

**Result**

- Segment accuracy ≈ 0.54.
- Track accuracy: 2/6 (both Bihag correct; Yaman and Kedar often predicted as Bihag).
- KNN confusion: strong Bihag cluster; other ragas pulled toward it.

**Notes**

- Confidence→voicing API warning acknowledged but not yet fixed.
- t‑SNE not yet run.

---

## 2026‑06‑16 – Segment‑level baseline v1b (Bihag/Bhoopali/Kedar)

**Goal**

- Balance concert lengths; examine Bhoopali instead of short Yaman tracks.

**Setup**

- Ragas: Bihag, Kedar, Bhoopali.
- Tracks:
  - Raag_Bihag: 27, 81
  - Raag_Bhoopali: 83, 105
  - Raag_Kedar: 8, 84
- Same segment settings.
- 608 valid segments.

**Result**

- Segment accuracy ≈ 0.56.
- Track accuracy: 4/6 (both Bihag and Bhoopali correct; both Kedar misclassified as Bhoopali).
- t‑SNE (`tsne_segments.png`):
  - Bihag and Bhoopali form reasonably coherent clusters.
  - Kedar segments lie between/inside these clusters.
- NOTE: Terminal output saved as segment_3_raag_result.txt


**Interpretation**

- At segment level, this feature set captures strong “Bihag/Bhoopali‑ness”.
- Kedar behaves like a bridge raga in this space.

**Next planned steps**

- Inspect “Bhoopali‑like” Kedar segments via NN‑based helper and listen to those windows.
- Optionally try overlap: hop < segment_length (e.g., 15 s).

## 2026‑06‑18 – Planning next raga ID experiments

**Goal**

- Plan the next phase of raga classification experiments beyond the current 3‑raga subset.

**Discussion / Decisions**

- Current priority is to improve raga ID accuracy before building any UI.
- Agreed to:
  - Expand from 3 to ~5–6 ragas, each with at least two long tracks, to avoid overfitting to the current subset.
  - Keep KNN as the baseline classifier for now and first refine the feature set (drop clearly noisy features, possibly add a few musically motivated ones).
  - Only after that, compare alternative classifiers (e.g., linear models, random forests) on the larger raga set.
  - Treat segment timing tweaks (different window/hop) as a secondary axis, to be explored after the above.

- Chosen ragas and tracks for the 6‑raga segment experiment:
  - Raag_Bihag: 27_Raag_Bihag, 81_Raag_Bihag
  - Raag_Kedar: 8_Raag_Kedar, 84_Raag_Kedar
  - Raag_Bhoopali: 83_Raag_Bhoopali, 105_Raag_Bhoopali
  - Raag_Abhogi: 20_Raag_Abhogi, 44_Raag_Abhogi
  - Raag_Shree: 0_Raag_Shree, 37_Raag_Shree
  - Raag_Lalit: 10_Raag_Lalit, 33_Raag_Lalit, 104_Raga_Lalit_-_Khayal

- Implemented a 6‑raga segment‑level KNN baseline:
  - Segment_length = 30 s, hop = 20 s, min_duration = 15 s.
  - Built ~600+ segments across all tracks, using the existing 52‑dim Stage‑2 feature vector.
  - Ran leave‑one‑track‑out evaluation and logged track‑level and segment‑level accuracy.

- Diagnostics run today:
  - t‑SNE of segment features (`tsne_segments.png`) shows reasonably coherent clusters for Bihag, Bhoopali, Abhogi, Shree; Kedar and Lalit segments lie in overlap regions.
  - Computed a 6×6 segment‑level confusion matrix:
    - Bhoopali, Bihag, Shree have majority of segments correctly classified.
    - Kedar and Lalit are heavily confused with Bihag/Bhoopali/Shree.
    - Abhogi segments often pulled toward Bhoopali.
  - Added helpers to print the confusion matrix and inspect nearest‑neighbour compositions for specific ragas/segments.

**Next steps (short term)**

- Use the neighbour‑inspection helper to analyse “least self‑like” segments for Kedar, Lalit, and Abhogi and listen to selected windows.
- Begin designing small feature tweaks informed by these confusions (e.g., emphasising swara patterns or intervals that distinguish Kedar/Lalit/Abhogi from Bihag/Bhoopali/Shree).

## 2026‑06‑19 – Feature subset refinement for 6‑raga segment KNN

**Goal**

- Inspect the current Stage‑2 feature set used per segment and define a smaller, musically sensible subset for the 6‑raga segment classification experiment.

**What was reviewed**

- Checked the actual current files used in the pipeline:
  - `raga_features.py`
  - `stage1_schema.py`
  - `segment_dataset.py`
  - `test_segment.py`
- Confirmed that segment features are controlled in `extract_raga_features_from_stage1(...)` inside `raga_features.py`.
- Confirmed that the current segment representation is built from:
  - tonic features,
  - swara proportion / count features,
  - relative pitch range features,
  - pitch histogram features.

**Feature discussion**

- Reviewed the usefulness of the Stage‑2 groups:
  - Relative range: `range_span_cents` considered more useful than `min/max/median` individually.
  - Histogram: `hist_peak_1_*`, `hist_entropy`, and `hist_concentration` considered the most useful; `hist_peak_2_*` and `hist_peak_3_*` considered weaker/noisier.
  - Tonic: `tonic_pc_cents` and `n_voiced_used` / `log_n_voiced_used` considered less useful for the current classification goal.
  - Swara features were kept intact; no swara was dropped for musical reasons.

**Feature subset chosen**

- Dropped from the feature vector:
  - `tonic_pc_cents`
  - `n_voiced_used`
  - `log_n_voiced_used`
  - `min_relative_cents`
  - `max_relative_cents`
  - `median_relative_cents`
  - `hist_bin_size_cents`
  - `hist_peak_2_cents`
  - `hist_peak_2_height`
  - `hist_peak_3_cents`
  - `hist_peak_3_height`

- Kept:
  - remaining tonic features,
  - all swara proportion features,
  - all log swara count features,
  - swara summary features,
  - `range_span_cents`,
  - `hist_ref_hz`,
  - `hist_peak_1_cents`,
  - `hist_peak_1_height`,
  - `hist_entropy`,
  - `hist_concentration`.

**Implementation**

- Added `DROP_NAMES` and `apply_feature_subset(...)` in `raga_features.py`.
- Applied the subset inside `extract_raga_features_from_stage1(...)`, so the reduced feature vector is now used automatically by downstream segment experiments without changing `segment_dataset.py` or `test_segment.py`.

**KNN result after subset**

- Re-ran the 6-raga segment KNN experiment using the reduced feature set.
- Observed:
  - track-level accuracy: 0.462
  - segment-level accuracy: 0.444
- Compared with the earlier full-feature version, this looked like a small improvement rather than a degradation.
- The reduced feature set therefore appeared to remove low-value noise without harming the useful structure of the segment representation.

**CSV export and feature-name verification**

- Exported the reduced segment feature dataset to CSV form as `key_segment_features_table.csv`.
- Confirmed that the CSV contains `raga_label` plus 41 numeric feature columns corresponding to the post-subset feature vector.
- Printed and matched the feature index mapping for columns `0` to `40`, confirming the retained feature list now includes:
  - tonic / target features,
  - swara proportions,
  - log swara counts,
  - voiced/confident/unassigned frame summaries,
  - `range_span_cents`,
  - histogram summary features.

**CSV-based feature analysis**

- Used the CSV to inspect which feature groups vary most across the 6-raga dataset.
- The strongest discriminative signal appeared to come mainly from swara proportion and log-count features rather than from the diagnostic frame-count features.
- This supported the earlier decision to prune weaker range / histogram detail features while retaining the musically interpretable swara-based representation.
- The remaining errors appeared to reflect genuine overlap in the current feature space rather than an obvious feature-selection mistake.

**Alternative classifiers tested**

- Since KNN accuracy remained modest after feature cleanup, tested two alternative classifiers directly on the exported CSV:
  - multinomial logistic regression,
  - random forest classifier.
- A helper script was prepared to:
  - load the CSV,
  - rename feature columns using the verified feature mapping,
  - train both models,
  - save detailed evaluation reports into separate text files instead of printing everything to the terminal.

**Logistic regression result**

- Logistic regression achieved segment-level accuracy of `0.8035`.
- This was a large improvement over the KNN result on the same feature table.
- Interpretation:
  - the current feature set is not random or weak,
  - it already contains strong class information,
  - and that information is sufficiently organized that even a linear classifier can separate much of the data.

**Random forest result**

- Random forest achieved segment-level accuracy of `0.9051`.
- This substantially outperformed both KNN and logistic regression on the CSV-based evaluation.
- Interpretation:
  - the current feature set contains useful non-linear structure,
  - and a tree-based ensemble is able to exploit interactions and thresholds that KNN and linear decision boundaries are not capturing well.

**Important evaluation note**

- These logistic regression and random forest runs used a stratified segment-level split on the CSV.
- They did **not** use a group-aware track-level split, because the exported CSV did not yet include a parent track identifier column for grouped evaluation.
- Therefore these accuracy values should be treated as optimistic segment-level estimates, not final track-safe benchmark numbers.
- Even so, the comparison was still useful because it strongly suggested:
  - the feature subset is solid,
  - classifier choice matters a lot,
  - and random forest is a promising next baseline.

**Artifacts created**

- `classifier_compare.py`
- `logistic_regression_analysis.txt`
- `random_forest_analysis.txt`

**Conclusion for the day**

- The feature-subset refinement was successful: it simplified the representation without hurting performance.
- KNN remained limited even after feature cleanup, suggesting the bottleneck was no longer just feature noise.
- Logistic regression showed that the feature table carries strong signal.
- Random forest showed that non-linear modelling may be much better suited to this segment representation than KNN.

**Next step**

- Add a parent `track_id` or equivalent grouping field to the exported segment CSV.
- Re-run logistic regression and random forest with a group-aware split so that segments from the same track do not appear in both train and test.
- If random forest still outperforms KNN under grouped evaluation, promote it as the main next classifier baseline for segment-level experiments.

## 2026-06-22 – Proper grouped evaluation and classifier comparison for 6-raga segment classification

**Goal**

- Move from an optimistic segment-level evaluation to a proper leakage-safe evaluation for the 6-raga segment classification experiment.
- Compare stronger classifiers against the earlier KNN baseline under this stricter setup.

**What changed in evaluation**

- Identified that the earlier CSV-based classifier comparison used a random segment split, which allows segments from the same track to appear in both train and test.
- Recognized this as an optimistic evaluation setup for the current problem, since the real task is to classify unseen tracks, not unseen segments from already-seen tracks.
- Switched to a **group-aware evaluation** using `track_id` as the grouping variable and `LeaveOneGroupOut`, so that in each fold one full track is held out for testing and no segments from that track are seen during training.[web:225][web:177]
- This makes the evaluation much stricter and more realistic, and avoids train/test contamination caused by within-track overlap.[web:178][web:176]

**Models evaluated**

- Kept the earlier KNN result as the reference grouped baseline:
  - track-level accuracy: 0.462
  - segment-level accuracy: 0.444
- Implemented and ran two additional grouped baselines on the same 6-raga setup:
  - multinomial logistic regression
  - random forest
- The evaluation was done on the segment dataset built from the same Stage-1 / Stage-2 pipeline, with track-level majority voting over segment predictions for final track classification.

**Main results**

- **Random forest**
  - track-level accuracy: 0.692
  - segment-level accuracy: 0.435
- **Logistic regression**
  - track-level accuracy: 0.538
  - segment-level accuracy: 0.394
- Compared to KNN, random forest gave a clear improvement in **track-level** accuracy while maintaining a similar segment-level accuracy.
- This suggests that random forest is currently the strongest classifier among the tested baselines for the present feature space and dataset.

**Observed confusion structure**

- Bhoopali, Shree, and Bihag are relatively stable under grouped evaluation and are usually identified correctly at the track level.
- Abhogi remains the weakest raga in the current setup and is heavily confused with Bhoopali and Kedar.
- Kedar is still mixed with Bhoopali and Abhogi.
- Lalit remains entangled with Bihag and Shree, especially for the large khayal track.
- These broad confusion patterns remain consistent across KNN, logistic regression, and random forest, indicating that the main limitation is not only the classifier but also the current feature representation and the small number of tracks per raga.

**Interpretation**

- The previous high classifier scores from random segment splits were not trustworthy as final results, because they were inflated by within-track similarity leaking into both train and test.[web:178][web:232]
- The grouped results are lower, but they are the correct numbers to use going forward because they reflect generalization to unseen tracks.[web:225][web:177]
- The classifier comparison is still useful: random forest improves over KNN and logistic regression, so classifier choice does matter, but only up to a point.

**Next development direction**

- Decided that further progress is unlikely to come from repeated classifier tuning on the same feature set.
- The next useful direction is to improve the representation extracted from the pitch contour, especially with features that capture:
  - local contour movement,
  - stable-note versus transition structure,
  - simple swara sequence patterns.
- Planned additions to the Stage-2 feature extractor:
  1. contour movement / slope features,
  2. stability-transition features,
  3. simple swara bigram / trigram pattern features.
- Also identified that Stage-1 must be extended to retain frame-level normalized pitch information (`relative_cents`, `voiced_mask`, etc.) inside `pitch_summary`, because these new feature families cannot be computed from only the current range and histogram summaries.

**Current status**

- The 6-raga experiment now has:
  - a reduced feature subset,
  - a proper grouped evaluation protocol,
  - and a stronger baseline classifier (random forest).
- This is now a valid proof-of-concept baseline for the first raga-identification stage, while also clearly showing where the present representation fails.

## 2026-06-23 – Adding contour-based features and testing longer segments

**Goal**

- Enrich the Stage-2 feature representation with basic contour and pattern information derived from the pitch contour, and check whether this improves 6-raga segment classification under proper leave-one-track-out evaluation.
- Test whether using longer segments (60 s instead of 30 s) changes performance in a meaningful way.

**New Stage-1 schema changes**

- Extended `pitch_summary` in `stage1_schema.py` to include frame-level tonic-normalized pitch information from `normalized_result`:
  - `times`
  - `voiced_mask`
  - `relative_cents`
  - `relative_cents_folded`
- These are now available directly in `stage1["pitch_summary"]`, not only inside the optional `artifacts`, so that Stage-2 feature extraction can compute contour-based features from them.

**New Stage-2 feature families**

Added three new feature families in `raga_features.py`, each implemented as a separate extractor returning `(values, names)` so they can be appended just like the existing tonic/swara/range/histogram blocks:

1. **Contour movement / slope features**
   - Compute frame-wise differences on the tonic-normalized pitch track.
   - Features include:
     - mean absolute pitch difference per frame,
     - standard deviation of pitch differences,
     - mean positive and negative pitch differences,
     - fractions of rising, falling, and near-flat frames,
     - mean step size for “non-flat” steps.
   - Aim: capture how much the contour moves within a segment and the balance between ascending, descending, and relatively stable motion.

2. **Stable vs transition region features**
   - Identify “stable” regions as spans where frame-wise pitch changes stay below a small threshold for a minimum number of frames, and treat the rest as “transition” regions.
   - Features include:
     - number, mean length, and max length of stable regions,
     - ratio of frames in stable regions vs total,
     - number, mean length, and max length of transition regions,
     - ratio of frames in transition regions vs total.
   - Aim: give coarse information about how much time a segment spends on sustained notes versus moving between them.

3. **Simple swara pattern (n-gram) features**
   - Quantize `relative_cents` to the canonical swara set to obtain a swara sequence for the segment.
   - Compute normalized counts for a small fixed set of swara bigrams and trigrams (e.g. Sa→Re, Re→Ga, Ga→Ma, … and Sa–Re–Ga, Ga–Ma–Pa, etc.).
   - Features: proportions of these selected bigrams and trigrams within the segment.
   - Aim: capture very simple, global statistics of characteristic step patterns without doing full motif discovery.

These new feature families are appended after the existing histogram features in `extract_raga_features_from_stage1(...)`.

**Grouped evaluation with new features (30 s segments)**

- Re-ran the 6-raga, leave-one-track-out experiment with the extended feature vector (old features + the three new families) and the same 30 s segments.
- Results:
  - **Random forest**
    - Track-level accuracy: 0.615
    - Segment-level accuracy: 0.384
  - **Logistic regression**
    - Track-level accuracy: 0.615
    - Segment-level accuracy: 0.375
- Compared to the earlier run (without the new feature families), both models’ accuracies decreased slightly and RF is only marginally ahead of logistic regression.
- The main confusion patterns remained essentially the same:
  - Bhoopali, Shree, and Bihag continue to be relatively well-identified at the track level.
  - Abhogi still has almost zero recall and is consistently misclassified as Bhoopali or Kedar.
  - Kedar remains heavily mixed with Bhoopali and Abhogi.
  - Lalit continues to form a triangle of confusion with Bihag and Shree, with individual tracks tipping toward different neighbours.
- The additional contour-based scalar summaries did not produce a clear improvement; they seem to slightly reshuffle errors without resolving the structurally ambiguous cases.

**Experiment with longer segments (60 s)**

- Re-built the segment dataset using 60-second segments (with the same general overlap logic) and re-ran the same grouped evaluation for both classifiers.
- Results were effectively unchanged compared to 30-second segments:
  - **Random forest**
    - Track-level accuracy: 0.615
    - Segment-level accuracy: 0.384
  - **Logistic regression**
    - Track-level accuracy: 0.615
    - Segment-level accuracy: 0.375
- Conclusion: simply doubling segment length does not materially change model performance or the confusion structure in this dataset, suggesting that the limiting factor is not segment duration but the combination of:
  - small number of tracks per raga,
  - and the use of aggregated, segment-level scalar features.

**Current understanding**

- The system now has:
  - a richer Stage-2 feature vector (including basic contour and pattern summaries),
  - an updated Stage-1 schema that exposes the normalized contour for feature extraction,
  - and a stable, leakage-free grouped evaluation protocol.
- Despite these additions, both logistic regression and random forest still struggle in a consistent way with Abhogi and Kedar, and with certain Lalit/Bihag/Shree cases.
- This reinforces the earlier conclusion that:
  - further small tweaks to scalar segment-level features and standard classifiers on this 13-track, 6-raga setup are unlikely to yield substantial gains, and
  - meaningful improvement will likely require either richer, more explicitly phrase-oriented representations, or a larger and more diverse dataset (or both).

## Transition to LLM Commentary Engine & Multimodal Grounding (Late June 2026)

Goal: Evaluate generative LLM architectures for high-level textual commentary generation (Target Persona: `advanced learner`), shifting from rigid scalar classifiers to a fluid musicological synthesis layer. Identify grounding gaps between raw audio signals, raw numerical matrices, and symbolic feature extraction.

---

## Experiment One: Pure Black-Box LLM Baseline (Zero-Grounding)

### Configuration
- **Model ID**: `gemini-2.5-flash`
- **Execution Script**: `run_baseline.py`
- **Inputs**: Raw audio waveform binary (`Raag_Bihag.mp3`) only.
- **Redactions**: Absolute removal of high-level text metadata (Raga name, artist identity, instrument type, structural boundaries).
- **Target Persona Prompt Configuration**: `role = "advanced learner"`

### Quantitative & Textual Observations

The execution ran successfully (`status: success`) on the test track `Raag_Bihag`. The complete text artifact logged in `experiment_one_blind_baseline.csv` evaluates as follows:

```text
"This compelling sarod rendition of Raga Yaman unfolds with remarkable depth and expressive power, tailored perfectly for an advanced learner appreciative of nuanced classical artistry.

The performance commences with the serene backdrop of the tanpura drone, setting an immediate tone of contemplation. The sarod enters with the unmetered *Alap*, meticulously constructing the raga's melodic architecture. The initial exploration is rooted in the lower octave (*mandra saptak*), patiently establishing the fundamental notes: Sa, a gracefully articulated *komal* Ga, a resonant *shuddha* Re, and the defining *teevra* Ma... [The narrative continues through an idealized textbook structure of Jor, Jhala, and a Teen Taal Bandish matching a classic instrument profile]."
```

### Conclusions & Grounding Gaps

**Abstract Hallucination**

The ungrounded model suffers a complete physical disconnection from the audio timeline. Lacking grounding tokens, its internal attention layer defaults to its strongest textual probabilistic patterns.

**Musicological Falsification**

The true track is a vocal/instrumental performance of Raag Bihag (utilizing a shuddha scale and teevra Ma transitions). The model confidently hallucinated Raag Yaman on the Sarod and invented a "gracefully articulated komal Ga" (a pitch interval that does not exist in Yaman grammar, which uses shuddha Ga).

**Linguistic Compliance**

The linguistic formulation is optimal. The script perfectly sustained advanced technical terms (mandra saptak, chikari, bol-vistar, sam) matching the advanced learner persona requirements, validating the narrative engine while exposing a factual void.

---

## Experiment Two: Raw Feature Grounding Baseline (Stage-1 MIR Assisted)

### Configuration

- **Model ID**: `gemini-2.5-flash`
- **Execution Script**: `run_experiment_two.py`
- **Inputs**: Raw audio waveform binary (`Raag_Bihag.mp3`) + External structural text metrics injected via prompt parameters:
  - `tonic_hz`: Extracted root frequency from `Raag_Bihag.ctonic.txt` (234.432110 Hz)
  - `pitch_contour_sample`: First 50 rows of raw float logs from `Raag_Bihag.pitch.txt`

### Quantitative & Textual Observations

The run executed successfully (`status: ok`), logging the following record inside `experiment_two_grounded_results.csv`:

```text
"From the outset, the sonic landscape is anchored by the unwavering presence of the Tanpura. The initial segments of the audio, where our Stage-1 MIR analysis records a melodic pitch contour of absolute zero, perfectly align with this establishing drone. This signifies a pure foundational state, where no active melodic line has yet emerged... As the performance progresses, a male vocalist embarks on an Alap... The Raga, as it gracefully reveals itself, exhibits the unmistakable characteristics of **Raga Bhairav**. The vocalist masterfully introduces the characteristic *komal Re* (approximately 248 Hz) and *komal Dha* (approximately 373 Hz)..."
```

### Conclusions & The "Zero-Array Trap"

**Data-Driven Contextual Hallucination**

Passing raw, un-aggregated time-series data tables introduced a unique multimodal error loop. Because the initial 50 rows of the pitch contour logged scientific notation zeroes (`0.000000000000000000e+00`), the model applied creative deduction. It argued that an extended sequence of absolute zero frequency values corresponded to an empty melodic canvas—specifically an opening Tanpura drone phase.

**Structural Re-centering**

Driven by the deduction of a long, heavy, meditative drone introduction, the model's musicological rules shifted to the Dhrupad genre accompanied by a Pakhawaj drum rather than a Tabla.

**Mathematical Coherence vs. Acoustic Truth**

The system achieved high interval precision: using the raw input 234.432110 Hz as $Sa$, it correctly calculated the acoustic intervals for a standard scale (e.g., Pa at $\approx 351$ Hz, komal Re at $\approx 248$ Hz). However, it mapped these perfect steps to Raga Bhairav (a morning raga), entirely inventing the presence of komal notes over a true Bihag (evening raga) track.

---

## Architectural & Package Refactoring (Module Integration)

To address path errors and module dependency friction when running scripts natively inside the `commentator/analysis/` folder, a formal package structure was enforced:

**Import Context Resolution**

Addressed `ImportError: attempted relative import with no known parent package` within `stage1_schema.py` by transitioning standalone execution to top-level module calls executed from the root workspace folder:

```bash
python -m commentator.analysis.experiment_three
```

**Path Nesting Resolution**

Fixed a duplicate directory crash (`[Errno 2] No such file or directory: .../saraga_hindustani/saraga1.5_hindustani/...`) caused by mirdata automatically appending the dataset name string to the user-defined `DATA_HOME`. The engine path resolution logic was refactored to check and isolate the parent folder context dynamically:

```python
env_data_home = os.environ.get("DATA_HOME", "~/mir_projects/data")
expanded_path = Path(os.path.expanduser(env_data_home))
data_home_dir = str(expanded_path.parent) if expanded_path.name == "saraga_hindustani" else str(expanded_path)
```

**Data Layer Abstraction**

Replaced direct `track_data.f0` array parsing with the project's native input/output loader wrapper (`commentator.io.loaders.load_track`), abstracting the instantiation of the custom `PitchContour` dataclass.

---

## Experiment Three: The Hybrid Feature-Summarized Framework

### Configuration

- **Model ID**: `gemini-2.5-flash`
- **Execution Script**: `experiment_three.py` (Run via root package module environment)
- **Inputs**: Raw audio binary stream + symbolic, high-level aggregate metrics computed across the complete pitch contour via `stage1_schema.py` and `swara_analyzer.py`:
  - `{tonic_hz}`: Verified baseline tuning frequency
  - `{dominant_swaras}`: Confidently mapped pitch nodes
  - `{least_used_swaras}`: Computed complement inverse array of omitted pitch intervals
  - `{min_cents}` / `{max_cents}`: Boundary ranges relative to $Sa$
  - `{basic_comment}`: The intermediate musicological summary string

### Quantitative & Textual Observations (Round Two Validation)

The framework was evaluated over target operational dataset tracks. The resulting matrix logged in `experiment_three_hybrid_results.csv` outlines three distinct edge-case behaviors:

**Case A: Complete Data Grounding (Track 27_Raag_Bihag)**

- **Extracted Features**: `tonic_hz: 146.4089`, `dominant_swaras: "Sa, Ga, Pa"`, `status: ok`
- **Generated Commentary Summary**:

```text
"The artist embarks on a slow, contemplative exposition... What is immediately striking, and confirmed by the aggregated MIR features, is the performance's profound reliance on a remarkably constrained set of dominant swaras: **Sa, Ga, and Pa**... The analysis of voiced pitch boundaries indicates a truly impressive vocal range, spanning from a minimum of -975.00 cents (deep into the *ati-mandra saptak*) to a maximum of 3645.03 cents (extending well into the *ati-taar saptak*)..."
```

**Case B: File System Integrity Mismatch (Track 72_Maru_Bihag)**

- **Status**: `error: FileNotFoundError`
- **Diagnostic Log**: `[Errno 2] No such file or directory: '.../Geetinandan : Part-3 by Ajoy Chakrabarty/Maru Bihag/Maru Bihag.pitch.txt'`
- **Analysis**: The native `saraga1.5_hindustani` index tracks a double audio file extension format (`.mp3.mp3`), introducing an annotation structural mismatch against the local disk directory hierarchy.

**Case C: Sub-Feature Extraction Mismatch (Track 81_Raag_Bihag)**

- **Extracted Features**: `tonic_hz: 196.5645`, `dominant_swaras: "Sa, Pa, ga"`, `status: ok`
- **Generated Commentary Summary**:

```text
"Notably, the performance strictly adheres to a highly specific and minimalist melodic framework... These include re, Re, Ga (shuddh gandhar), Ma (shuddh madhyam), Ma^ (tivra madhyam), dha, Dha, ni, and Ni (shuddh nishad). This deliberate melodic economy, focusing intently on Sa, Pa, and ga, creates a unique interpretation of the raga..."
```

---

## Current System Understanding & Research Findings

**Total Elimination of Random Drift**

Experiment Three confirms that feeding the linguistic engine high-level symbolic summaries completely boxes it in. The model no longer invents random instruments or unaligned raga scales out of probabilistic text habits.

**Robustness Under Data Constraints**

When the MIR pipeline returns placeholder values (as seen in early baseline tests with empty features), the prompt instructions successfully force the model to acknowledge the data limits, avoiding factual fabrications.

**The Core MIR Grounding Gap**

Track 81_Raag_Bihag highlights the current critical bottleneck of the system. While the LLM followed the prompt rules perfectly by listing shuddha Ga as omitted and focusing entirely on komal Ga (ga), the ground-truth raga is Bihag—which fundamentally relies on shuddha Ga as its defining structural pillar.

**Conclusion**

The error is no longer localized in the Stage-2 Language Layer; it tracks back to the Stage-1 `swara_analyzer.py` core, where instrumentation tracking artifacts (such as string bends, heavy vocal glides, or tracking noise) are mathematically mapped to adjacent microtonal bins, creating a musicological contradiction.

---

## Next Strategic Steps

- **Implement Exponential API Backoff**: Embed try-except handlers targeting `google.genai.errors.APIError (503)` to protect long dataset loops from temporary server capacity spikes.

- **Refine the Swara Analyzer Filtering**: Optimize the circular distance cents tolerance parameters inside `swara_analyzer.py` to screen out tracking artifacts, ensuring accidental notes do not override the true structural syntax of the raga before compiling the prompt.

- **Stage-3 Evaluation Protocol Setup**: Prepare the generated `experiment_three_hybrid_results.csv` dataset for formal blind evaluation by human musicologists to score commentary accuracy and description quality.