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

---

## 2026-08-14 to 2026-08-15 – Backfilling lost t-SNE snapshots and fixing the overwrite bug

**Problem found**

- `plot_tsne_segments(...)` and `inspect_features.py`'s `plot_tsne(...)` always saved to a fixed filename (`tsne_segments.png` / `tsne_plot.png`). Every re-run with a different raga subset, feature set, or segment window silently overwrote the previous image.
- Of the six distinct segment-level t-SNE configurations run over the project's history, only two happened to survive by being committed at the right moment; the rest were lost before ever being saved.

**Recovery**

- Two of the lost configurations (v1b: 3-raga Bihag/Bhoopali/Kedar; the first 6-raga run) used the pre-`DROP_NAMES`, pre-contour 52-feature extractor, so they were regenerated by checking out the codebase as it existed right after the original "segment wise feature set creation" commit (via a disposable `git worktree`, not by modifying the current code) and re-running the same track lists.
- A third lost configuration (the 60s/50s-hop variant of the current 71-feature extractor) was regenerated directly with today's code.
- All six snapshots (2 regenerated historical, 1 regenerated current-code, 3 already-preserved) were consolidated into `docs/experiments/tsne/`, with a `README.md` mapping each file to its date, raga set, feature count, and segment window.

**Correction to this log**

- Regenerating the v1b run surfaced a discrepancy: the 2026-06-16 entry above states "608 valid segments" at a non-overlapping 30s/30s hop, but the actual saved terminal output (`segment_3_raag_result.txt`) recorded 910 segments. 910 segments is only reproducible with a 30s/20s hop (confirmed by rerunning both ways) — so the real v1b run evidently used 30s/20s, not 30s/30s as originally logged. The archived plot and this note reflect the verified 910-segment/30s-20s version; the original entry is left as-is above rather than edited, per this log's append-only convention.

**Root-cause fix**

- Both plotting functions now derive their output filename from the run's raga count, feature count, and (for segments) window size instead of using a constant, so future runs no longer clobber each other. Verified with two different synthetic configs producing two distinct files.

---

## 2026-08-15 to 2026-08-16 – Scaling beyond Saraga: the CompMusic Raga Recognition Dataset (Dunya / HMD)

**Motivation**

- Under the proper grouped (`LeaveOneGroupOut`) evaluation established on 2026-06-22, the 6-raga/13-track Saraga subset caps out around 61.5% track-level accuracy (matching the 2026-06-23 entry above) for both logistic regression and random forest — consistent with sample scarcity (2-3 tracks per raga) rather than a feature or classifier problem. Discussed with Ajay Srinivasamurthy, who recommended scaling up via the larger Dunya/CompMusic dataset rather than continuing to tune the current 13-track setup.

**Dunya portal access attempt**

- Registration on the Dunya web portal (`dunya.compmusic.upf.edu`) failed silently, and the password-reset email never arrived. `pip install dunya` fails (no such PyPI package); the correct client is `compmusic` (`pip install compmusic`, `from compmusic import dunya`), which still requires an API token obtained through the same broken portal login (401 without it).
- Raised with Ajay, then with Thomas Nuttall (MTG) and Alastair Porter. Tom confirmed the backend/API-access owner (Cristina) is on vacation until the university reopens in September, and offered to share data manually in the meantime.

**Zenodo route (recommended by Ajay, verified directly)**

- `10.5281/zenodo.7278505` → *"Indian Art Music Raga Recognition Dataset (features)"*, a single 3.6 GB zip, freely downloadable with no login or token, CC-BY 4.0. Contains pitch/tonic/nyas/tani-segment annotations and metadata (MBIDs, raga IDs) for both the Carnatic (CMD: 480 recordings, 40 ragas) and Hindustani (HMD: 300 recordings, 30 ragas, 10 recordings/raga, 116 hours) collections.
- `10.5281/zenodo.7278510` → the matching audio, 9.2 TB combined across both traditions, access-restricted (requires a formal request citing purpose) — same category of gatekeeping currently blocking Dunya API access, just for a different resource. Not needed for the current classification pipeline, which only ever consumes pitch/tonic arrays (only `experiment_three.py`, the LLM commentary layer, touches raw audio).

**Key finding: no Dunya API needed for this**

- `mirdata` already ships a dedicated loader, `compmusic_raga` (`mirdata.initialize("compmusic_raga", ...)`), for this exact dataset. `dataset.download()` fetches the free features zip automatically with zero credentials (audio is deliberately excluded from the automatic download). Each track exposes `track.pitch` (an `F0Data` object with the same `times`/`frequencies`/`voicing` shape as `saraga_hindustani`), `track.tonic`, `track.raga`, and `track.tradition` (`"hindustani"` vs `"carnatic"`) — meaning the Dunya web API / `compmusic` client route is unnecessary for the pitch+metadata scaling goal.

> **Correction (2026-08-16, after actually running it):** the second half of this
> claim is wrong. The loader and the `tradition` field both exist, but mirdata's
> `compmusic_raga` **index v1.0 contains only the 477 Carnatic recordings** —
> `dataset.track_ids` yields *zero* Hindustani tracks, and every indexed track
> reports `tradition == "carnatic"`. The downloaded archive does contain both
> traditions; mirdata simply cannot see half of it. The headline conclusion
> (no Dunya API needed) still holds — the download works credential-free — but
> the Hindustani half must be read off the extracted tree directly. See the
> 2026-08-16 entry below.

**New test harness**

- Added `tests/run_compmusic_check.py`: initializes `compmusic_raga`, downloads the features archive, filters to Hindustani-tradition tracks, checks the raga distribution against the documented "10 recordings per raga," and wraps one sample track's pitch into the existing `PitchContour` dataclass and runs it through `build_stage1_schema(...)` — confirming (or breaking on) the claim that no Stage-1/Stage-2 code needs to change to consume this dataset. Not yet run.

**Next steps**

- Run `tests/run_compmusic_check.py`, confirm the 10-per-raga distribution and Stage-1/Stage-2 compatibility hold.
- Extend `commentator/io/loaders.py` with a `compmusic_raga`-based loader alongside the existing `saraga_hindustani` one.
- Re-run the segment/classifier comparison at Dunya scale (30 ragas, 10 tracks/raga instead of 6 ragas, ~2/raga) to see whether track-level accuracy moves past the current ~61.5% ceiling.

---

## 2026-08-15 to 2026-08-16 – Repository restructuring

Housekeeping pass to reduce friction while scaling up, in the same spirit as the "Architectural & Package Refactoring" entry above.

- **`test_*.py` naming collision**: several files under `tests/` and the repo root were named `test_*.py` without being actual pytest tests (no `test_` functions; some execute pipeline code at module level), which caused `pytest` to try to collect and execute them, crashing on missing local paths. Renamed every non-pytest driver script to `run_*.py` (`test_segment_lr_rf.py` → `run_segment_lr_rf.py`, `tests/test_segment.py` → `tests/run_segment_baseline_knn.py`, `tests/test_mirdata.py` → `tests/run_mirdata_check.py`, etc.), leaving `test_` reserved for `commentator/tests/test_pitch_contour.py`, the one real pytest test.
- **Generated artifacts consolidated**: feature CSVs, t-SNE plots, and classifier reports were scattered across the repo root and `tests/`. They now write into `outputs/` (`outputs/key_segment_features_table.csv`, `outputs/classifier_compare/`, etc.), resolved relative to each script's own file location rather than the caller's working directory — which also explains why two differently-shaped copies of `key_segment_features_table.csv` existed side-by-side in `tests/` and at the repo root prior to this cleanup (they were written by the same code run from two different working directories).
- **Stopped tracking `__pycache__/`/`*.pyc`** in git; added to `.gitignore`.

---

## 2026-08-16 – Running HMD for real, and discovering the tonic estimator is unreliable

Three connected findings: the HMD harness needed rewriting, the dataset scale-up
is confirmed viable, and the ground-truth tonics it exposes revealed a serious
problem in Stage-1 that also affects the existing Saraga results.

### 1. `tests/run_compmusic_check.py` — first real run

- **Initial failure was trivial and unrelated to the dataset**: `ModuleNotFoundError: No module named 'commentator'`. Running a script inside `tests/` puts `tests/` on `sys.path`, not the repo root. Fixed with an explicit `sys.path` bootstrap resolved from the script's own location. *Note: every other script under `tests/` shares this latent problem and only works when invoked with the repo root already importable.*
- **Download throughput**: mirdata's downloader pulled at ~100 kB/s (8–17 h ETA for the 3.44 GB archive). Zenodo throttles **per connection, not per client** — 8 parallel HTTP range requests achieved ~900 kB/s, completing in ~1 h. The reassembled file was verified against mirdata's expected md5 (`5dfc26dd1c2652ab75a62faec7f45f08`) before being handed over; mirdata then recognised it and skipped its own download. Worth remembering for any future large Zenodo fetch.
- **The mirdata index limitation** (see correction above): `Hindustani tracks: 0` on the first successful run. Confirmed by inspecting the shipped index JSON directly — 477 tracks, none Hindustani.

### 2. HMD is confirmed viable as a scale-up target

Rewritten to read `RagaDataset/Hindustani/` off the extracted tree:

```
Hindustani tracks: 300
Ragas: 30          (exactly 10 recordings per raga, all 30)
```

- **The documented HMD numbers hold exactly** — 300 recordings, 30 ragas, 10 each, versus the current 6 usable ragas at 2–3 tracks each.
- `.pitch` files are two-column TSVs (time_s, freq_hz; 0.0 = unvoiced) and `.tonic` a single float — both feed `PitchContour` → `build_stage1_schema(...)` with **zero pipeline changes**, confirming the original compatibility claim.
- **Tracks must be joined to metadata by MBID, not by path.** The dataset's own `_info_/path_mbid_ragaid.txt` records paths that do not match what is on disk for **64 of the 300** tracks (album-name mismatches). A path-based join silently loses 21% of the dataset; the MBID suffix present on every feature filename resolves all 300.

### 3. Tonic estimation validated against ground truth — and it fails often

HMD ships an annotated tonic per recording, which prompted checking whether
**Saraga does too. It does** — `track.tonic` (from `ctonic_path`) has been
available on every track all along, and the pipeline has never used it,
estimating the tonic instead. So the estimator's error rate had never been
measured on any dataset.

Added `tests/run_tonic_validation.py`, which compares estimated vs annotated
tonic at two levels. Segment level is the one that matters: the classifier
calls `build_stage1_schema` **per 30 s window**, so every segment carries its
own independently estimated tonic.

Results on the 13-track / 6-raga experiment set (n=1809 segments — matching the
segment count in `segment_6_raag_result.txt`, i.e. the same data the 0.9051
figure was computed on):

| | track level (n=13) | segment level (n=1809) |
|---|---|---|
| within ±50 cents (raw) | 46.2% | **26.4%** |
| within ±50 cents (octave-folded) | 61.5% | **49.8%** |
| pitch-class errors | 38.5% | **50.2%** |
| median abs. error (folded) | 5.0 cents | 75.0 cents |

Why the two columns differ: swara assignment folds cents mod 1200
(`swara_analyzer.py`), so a **pure octave error is largely harmless** to
`swara_prop_*` (though it still corrupts `tonic_hz`, `log_tonic_hz`, and the
relative-cents range features). A **pitch-class error shifts every swara
assignment** and invalidates the whole vector.

- **Half of all training segments (50.2%) have a pitch-class-wrong tonic.** Median folded error of 75 cents means these are wrong scale degrees, not near-misses.
- Track-level failures are musically coherent, the signature of a histogram estimator locking onto the wrong scale degree: `8_Raag_Kedar` +695¢ (a perfect fifth — Pa read as Sa), `84_Raag_Kedar` +495¢, `44_Raag_Abhogi` −485¢, `10_Raag_Lalit` +405¢, `81_Raag_Bihag` −305¢.
- Errors are **concentrated per track** (`81_Raag_Bihag` 81.7% of segments, `44_Raag_Abhogi` 70.2%, `10_Raag_Lalit` 68.4%), and correlate with raga — both Kedar tracks fail; both Bhoopali and both Shree pass.

**Implication for the 0.9051 RF baseline.** This does *not* show the number was
measured incorrectly — the grouped `LeaveOneGroupOut` protocol is sound. But a
*consistent* wrong tonic offset within a track still yields consistent features,
so the classifier may be keying on recording-specific estimator artifacts
("Kedar's tonic reads a fourth sharp") rather than raga structure. That would
score well under leave-one-track-out on 2–3 tracks per raga while failing to
generalise to HMD's 10 recordings per raga. Untested either way as of this entry.

*Caveats:* annotations are treated as ground truth without independent
verification (the octave-error cases in particular could reflect annotation
octave ambiguity), and the HMD spot-check used the first recording of each raga
rather than a random sample.

### 4. `commentator/io/` restructured into per-dataset adapters

Driven by a concrete problem: the HMD loading code was living in
`tests/run_compmusic_check.py`, and the first script written against it had to
`sys.path.insert` into `tests/` and import from a test script.

- `commentator/io/loaders.py` → **`saraga.py`** (`git mv`, history preserved), plus a `SaragaHindustani` adapter and a new `get_tonic_for_track(...)`.
- New **`commentator/io/compmusic.py`** holding the HMD loading logic and the reasoning about mirdata's index.
- Both expose the same interface — `name`, `list_tracks()`, `get_pitch(track_id)`, `get_tonic(track_id)` — so pipeline code works against either without changes. `build_segment_feature_dataset(tracks, get_pitch_fn, ...)` already took a callback, so no analysis code needed to change.
- Deliberately **not** added: a `Protocol` base class, a `get_dataset(name)` factory, or a `loaders.py` compatibility shim — all speculative abstraction at two datasets and five call sites. Existing module-level function names are unchanged, so only import lines moved (7 files).
- `tests/run_compmusic_check.py` dropped from ~200 to ~85 lines. Verified: all touched files compile, `pytest` still collects exactly 1 real test, the smoke test still passes, and a single dataset-agnostic function runs against both adapters.

### 5. Re-running with annotated tonics — and finding the 0.9051 was leaky

Added an optional tonic override: `tonic_hz` on `analyze_pitch_musically` /
`build_stage1_schema`, and `get_tonic_fn` on `build_segment_feature_dataset`,
exposed as `run_segment_lr_rf.py --annotated-tonic`. The histogram is still
computed and only the tonic is replaced, so histogram-derived features are
unchanged and the comparison isolates the tonic variable; the value that would
have been estimated is kept as `tonic_result["estimated_tonic_hz"]`. Output
filenames gain an `_annotated-tonic` tag, since both runs otherwise produce
identical raga/feature/window counts and would overwrite each other.

**First finding: 0.9051 was never a grouped result.** Re-running the baseline
with estimated tonics reproduces the ~61.5% track-level ceiling from the
06-22/06-23 entries — RF segment-level is **0.384, not 0.9051**. The two numbers
come from different protocols. `build_segment_feature_dataset` writes
`key_segment_features_table.csv` as `raga_label` plus bare numeric columns with
**no `track_id`**, so `classifier_compare.py`'s `detect_group_column()` returns
`None` and falls back to `train_test_split(..., stratify=y)` — a random split
over segments. Segments are 30s windows at 20s hop, so consecutive segments
overlap by 10s and near-duplicate windows land in both train and test. The
script reports this honestly ("Group-aware split used: no; stratified segment
split"); the group column simply never reached the CSV. **0.9051 and 0.8035
should not be cited.**

Quantified on the actual data (standardised feature space):

| segment pair | mean distance |
|---|---|
| adjacent segments, same track (10s overlap) | **7.25** |
| distant segments, same track | 10.76 |
| same raga, **different track** — the real task | **11.19** |

Under the random split, **94.9% of test segments have an overlapping neighbour
in the training set**, and **all 13 tracks appear on both sides**. So 0.9051 was
measuring the 7.25 problem while the question of interest is the 11.19 one.

**Fix applied**: `build_segment_feature_dataset` now exports `track_id` in
`key_segment_features_table.csv` (as a string, so consumers selecting numeric
columns as features cannot pick it up as an input; `segment_index` is
deliberately *not* exported, since it is numeric and would be swept up as a
meaningless positional feature). `detect_group_column()` now finds it and
`GroupShuffleSplit` engages — the report line changes from "no; stratified
segment split" to "yes (track_id)". Re-running the identical flow, estimated
tonic:

| `classifier_compare.py` | before (leaky) | after (grouped) |
|---|---|---|
| Logistic Regression | 0.8035 | **0.2402** |
| Random Forest | 0.9051 | **0.2677** |

The leak accounted for essentially the entire figure. These land *below* the
`LeaveOneGroupOut` numbers (0.375/0.384) because this is a harsher protocol: a
single `GroupShuffleSplit` holds out ~4 whole recordings at once and trains on
the remaining ~9, whereas leave-one-track-out trains on 12 and averages over 13
folds. `run_segment_lr_rf.py`'s numbers remain the ones to quote.

Running the same grouped flow on the annotated-tonic CSV gives LR **0.2500**
and RF **0.2717**, versus 0.2402 / 0.2677 estimated — i.e. under *this*
protocol the tonic fix is worth ~1 point and is **within noise**, unlike the
+0.145 it is worth under `LeaveOneGroupOut`. The two are not in conflict: a
single `GroupShuffleSplit` on 13 recordings is a high-variance estimate (one
draw, ~9 training tracks, so some ragas contribute only a single track), while
the leave-one-track-out figure averages 13 folds over all 1809 segments. The
`LeaveOneGroupOut` result is the more reliable of the two, but this is a useful
reminder that the tonic improvement has been demonstrated under one protocol,
not two.

> **Correction to section 3 of this same entry:** the caveat written earlier
> today attributed the unreliability of 0.9051 primarily to the tonic problem.
> That is wrong about the main cause — the dominant factor is the non-grouped
> split described above. The tonic issue is real and independent, but it is not
> what inflated 0.9051 to 0.90.

**Second finding: annotated tonics genuinely help.** Same 1809 segments, 71
features, `LeaveOneGroupOut`:

| | estimated tonic | annotated tonic | delta |
|---|---|---|---|
| LR track-level | 0.615 | 0.615 | — |
| LR segment-level | 0.375 | **0.520** | +0.145 |
| RF track-level | 0.615 | 0.538 | −0.077 |
| RF segment-level | 0.384 | **0.441** | +0.057 |

- Segment-level improves for both classifiers — the clearer signal, at n=1809.
- Track-level moves are one track either way (13 tracks = 7.7% each: 8/13 vs 7/13) and should not be read as meaningful.
- **The LR/RF ranking flips.** RF led under both the leaky split (0.9051 vs 0.8035) and the grouped estimated-tonic run (0.384 vs 0.375); with correct tonics LR leads clearly (0.520 vs 0.441). Consistent with RF's earlier advantage coming partly from carving out track-specific estimator artifacts that a linear model could not exploit — exactly the failure mode predicted in section 3.
- Absolute accuracy remains low, as expected: 6 ragas at 2–3 tracks each. Fixing the tonic removes a confound; it does not solve sample scarcity. That is what HMD is for.

### Next steps
- **Write `track_id` into `key_segment_features_table.csv`** so `classifier_compare.py` can actually group by track, closing the leak at its source rather than relying on people knowing not to trust that flow.
- Investigate the tonic estimator itself (octave/fifth resolution in `resolve_tonic_octave`); the override added here is a workaround that only helps where an annotation exists.
- **Scale to HMD (30 ragas × 10 recordings) with annotated tonics throughout.** With the tonic confound removed and the leaky number retired, sample scarcity is now the clear remaining bottleneck, and HMD addresses exactly that.

---

## 2026-08-17 – HMD pilot runs: sample scarcity confirmed as the bottleneck

Two pilots before committing to the full 300-track run. Both use the existing
Stage-1/Stage-2 pipeline unchanged; the only additions were dataset selection
and subsetting flags on `run_segment_lr_rf.py`, a `run_tag` discriminator on
generated filenames, and a `.tonicFine` preference in the HMD adapter.

### Method

| | Pilot 1 | Pilot 2 |
|---|---|---|
| Purpose | plumbing, timing, failure rate | controlled scale-up test |
| Tracks | 30 (1 per raga) | 50 (10 per raga) |
| Ragas | all 30 | 5: Bihāg, Kēdār, Bhūp, Ābhōgī, Śrī |
| Tonic | estimated | estimated **and** annotated |
| Evaluation | none (see below) | `StratifiedGroupKFold(10)`, grouped by `track_id` |
| Segments | 2,077 | 3,631 |

Pilot 2's five ragas are deliberately five of the six in the Saraga experiment
set (`Bhoopali` = `Bhūp`, `Shree` = `Śrī`; `Lalit` dropped as the least
comparable, being the only Saraga raga with three recordings). This makes it a
controlled comparison rather than a fresh benchmark: same ragas, same features,
same 30s/20s window, same pipeline — the variable is **10 recordings per raga
instead of 2–3**.

Pilot 1 runs no classifier by design: with one recording per raga, a
group-aware split cannot both train and test on a class. It also means Pilot
1's t-SNE cannot distinguish raga identity from recording identity — the two
are perfectly confounded — so that plot shows only that extraction works.

`StratifiedGroupKFold(10)` was chosen over `LeaveOneGroupOut` for HMD: with
exactly 10 recordings per raga it holds out precisely one recording of each
raga per fold, and costs 10 model fits instead of 50–300.

### Pilot 1 results — extraction is clean and cheap

| metric | value |
|---|---|
| Valid segments | 2,077 |
| Failed segments | **0 (0.00%)** |
| Feature dimension | 71 (identical to Saraga) |
| Build time | 95.5 s |
| Per segment / per track | 0.046 s / 3.18 s |
| **Projected full 300-track build** | **~16 min** |

Zero failures across 30 previously unseen recordings, and an identical feature
vector shape, confirm the adapter and Stage-1 need no HMD-specific handling.
The earlier estimate of ~1 hour per configuration was pessimistic by ~4x, which
makes running the full set in both tonic modes cheap.

One structural difference from Saraga: HMD segment counts per track are highly
uneven (min 14, median 67, max 157; durations ~5–52 min). Segment-level
accuracy is therefore dominated by the longest recordings, while track-level
accuracy weights recordings equally.

### Pilot 2 results — large gains, and the tonic effect confirmed at scale

`StratifiedGroupKFold(10)`, 50 tracks, 3,631 segments, chance = 0.2000:

| tonic | model | track acc | segment acc | segment × chance |
|---|---|---|---|---|
| estimated | Logistic Regression | 0.9400 | 0.6673 | 3.3× |
| estimated | Random Forest | 1.0000 | 0.8160 | 4.1× |
| **annotated** | **Logistic Regression** | **1.0000** | **0.9391** | **4.7×** |
| **annotated** | Random Forest | 1.0000 | 0.9303 | 4.7× |

**The annotated tonic is worth far more here than it was on Saraga**: +0.272
segment accuracy for LR (0.6673 → 0.9391) and +0.114 for RF, on 50 recordings
across 10 folds rather than 13 recordings. The 2026-08-16 entry noted the tonic
gain was demonstrated under only one protocol and was within noise under a
single `GroupShuffleSplit`; this is the independent confirmation that was
missing.

The LR/RF ranking behaves exactly as predicted on 2026-08-16. With estimated
tonics RF leads substantially (0.8160 vs 0.6673) — consistent with RF
exploiting non-linear, recording-specific estimator artifacts. With correct
tonics the gap closes and reverses slightly (0.9391 vs 0.9303): once swara
proportions mean the same thing across recordings, the linear model suffices.

Segment-level confusion (RF, annotated) is musically sensible — residual error
concentrates between **Kēdār and Bihāg**, both Kalyan-family ragas using both
Ma variants, while Śrī (komal re) is nearly perfectly separated at 0.98.

### Comparison with Saraga — and why it is not apples-to-apples

| | Saraga (6 raga, 13 tracks, LOGO) | HMD Pilot 2 (5 raga, 50 tracks, SGKF-10) |
|---|---|---|
| best segment acc (annotated) | 0.520 | **0.9391** |
| × chance | 3.1× | **4.7×** |
| track acc | 0.615 | **1.0000** |

Differences that make this indicative rather than conclusive: 5 classes vs 6
(chance 0.200 vs 0.167 — hence the ×chance column), `StratifiedGroupKFold(10)`
vs `LeaveOneGroupOut`, and entirely different recordings. Even normalised for
chance the improvement is large, and track-level accuracy going from 8/13 to
50/50 is not explicable by the protocol change alone. **Sample scarcity was
indeed the binding constraint**, as Ajay suggested.

### Caveat: the album/session effect probably inflates Pilot 2

100% track accuracy warrants the same scepticism applied to 0.9051, so the
recordings were checked for shared provenance. **21 of the 50 tracks (42%)
share an artist *and* album with another recording of the same raga:**

| raga | independent (artist+album) sessions | recordings |
|---|---|---|
| Bihāg | **5** | 10 |
| Kēdār | 6 | 10 |
| Bhūp | 8 | 10 |
| Śrī | 8 | 10 |
| Ābhōgī | 9 | 10 |

Half of Bihāg's recordings come from one album (`Abdul_Rashid_Khan – Rasan_Piya
Volume 1`), and half of Kēdār's from `Volume 3` by the same artist. Grouping by
`track_id` correctly prevents *segment* leakage, but it does not prevent the
model from recognising a **recording session** — same artist, tanpura, tonic and
room — shared between train and test. This is the well-known album effect, and
it is the same class of confound as the 2026-08-16 leak, one level up.

The control is to group by `(artist, album)` instead of `track_id`, which
`run_segment_lr_rf.py --group-by album` now supports. The 50 recordings resolve
to **33 independent sessions**, and Bihāg's 5 cap the split at
`StratifiedGroupKFold(5)`.

### Album-grouped control — the result largely survives

Same 50 recordings, same features, same 3,631 segments; only the CV grouping
unit changes.

| tonic | model | grouped by track | grouped by album | Δ segment |
|---|---|---|---|---|
| estimated | LR | 0.6673 | 0.6502 | −0.017 |
| estimated | RF | 0.8160 | 0.7981 | −0.018 |
| **annotated** | **LR** | 0.9391 | **0.9193** | **−0.020** |
| annotated | RF | 0.9303 | 0.8882 | −0.042 |

Track-level under album grouping: LR 0.9600, RF 0.9200 (annotated); LR 0.8600,
RF 0.9600 (estimated).

**The album effect is real but small** — 2–4 points of segment accuracy, not
the collapse that would indicate the model was mostly recognising recording
sessions. And part of even that is not the album effect at all: album grouping
forces 5 folds instead of 10, so each model trains on ~40 rather than 45
recordings. The drop is therefore an upper bound on the confound.

Two further observations:

- **LR's lead over RF widens under the strictest grouping** (0.9193 vs 0.8882, against 0.9391 vs 0.9303 when grouped by track). Consistent with the pattern seen throughout: RF extracts more from recording-specific idiosyncrasy, so tightening the protocol costs it more. Every time the evaluation has been made stricter, RF has lost more than LR.
- The tonic effect is undiminished by album grouping: +0.269 segment accuracy for LR (0.6502 → 0.9193). It is not an artifact of the grouping choice.

**Headline for the 5-raga pilot, under the strictest protocol run so far:
segment 0.9193, track 0.9600, 4.6× chance** (LR, annotated tonic, album-grouped,
`StratifiedGroupKFold(5)`). This is the number to quote.

### Artifacts

All tagged so nothing overwrites prior runs:
`outputs/key_segment_features_table_hmd-pilot{1,2}-*.csv`,
`outputs/tsne_segments_hmd-pilot{1,2}-*.png` (with `_annotated-tonic` variants).

### Next steps

- **Full 300-track / 30-raga run**, annotated tonic, both groupings (done — see below).
- Point `run_tonic_validation.py` at HMD for all 300 tracks; only 20 were spot-checked, and the estimator error rate on HMD is still not properly characterised.
- The tonic estimator itself remains unfixed (octave/fifth resolution in `resolve_tonic_octave`). Annotated tonics are a workaround that will not transfer to data lacking annotations.

---

## 2026-08-17 – Full HMD run: 30 ragas, 300 recordings

The scale-up run the whole preceding sequence was building toward.

### Dataset statistics gathered first (bias audit)

Before running, per-raga structure was measured across all 300 recordings:

- **Total 116.1 hours / 20,821 segments**, matching the dataset's documented "116 hours" exactly.
- Track durations span **1.6 to 71.1 minutes** (median 21.4). Per-track segment counts span 5 to 213.
- Track counts are perfectly balanced (10 per raga) but **segment counts are not**: Śrī yields 1,095 segments, Dēś only 392 — a **2.8× imbalance**. Since the classifier trains on segments, long ragas carry ~3× the weight, and segment-level accuracy is dominated by them. **Track-level accuracy is the fairer headline metric on this dataset.**
- Artist diversity per raga ranges from **2 to 10**. The extreme case is **Khamāj: 10 recordings from just 2 artists across 4 albums**.
- Across all 300 recordings there are only **158 independent (artist, album) sessions** — roughly half the collection shares a session with another recording, more overlap than the 5-raga pilot's 33/50.

### Method

Annotated (`.tonicFine`) tonic, 30s/20s windows, 71 features. Features are
identical across grouping schemes — grouping changes only cross-validation —
so extraction runs once and both evaluations reuse it (`--group-by track album`),
writing separate result files. Extraction: **20,821 segments, zero failures,
895 s (14.9 min)**, matching Pilot 1's ~16 min projection.

### Results

| grouping | CV | model | track acc | segment acc | × chance |
|---|---|---|---|---|---|
| track | SGKF(10), 300 groups | **Logistic Regression** | **0.9300** | **0.7079** | **21.2×** |
| track | SGKF(10), 300 groups | Random Forest | 0.9333 | 0.6735 | 20.2× |
| album | SGKF(4), 158 groups | **Logistic Regression** | **0.8633** | **0.6648** | **19.9×** |
| album | SGKF(4), 158 groups | Random Forest | 0.8533 | 0.6264 | 18.8× |

Chance = 0.0333. Misclassified recordings: 21/300 (LR) and 20/300 (RF) under
track grouping; 41/300 and 44/300 under album grouping.

The album-grouped fold count was **automatically capped at 4 by Khamāj**, which
has only 4 independent sessions. As in the pilot, part of the track→album gap
is therefore reduced training data (~75% vs ~90% per fold), not the album
effect alone — so **4–8 points is an upper bound** on the session confound. It
is larger than the pilot's 2–4 points, consistent with the full set having
proportionally more session overlap.

### The Khamāj hypothesis was wrong

The 2026-08-17 pilot entry predicted Khamāj would score conspicuously *high*,
since 10 recordings from 2 artists invites the model to recognise a voice
rather than the raga. **It did not.** Khamāj's F1 is **0.598** under track
grouping, below the 0.696 macro average, and **0.523** under album grouping. It
does lose more than average when sessions are separated (−0.075 vs −0.048
macro), so it is more session-dependent than most — but the artist-identity
confound did not inflate it. Keeping it in the study was correct.

### What does predict per-class performance: data volume

Correlation between segments per raga and per-class F1 is **+0.38**. The
bottom-10 classes average 545 segments, the top-10 average 743.

| | weakest | strongest |
|---|---|---|
| | Dēś F1 0.488 (392 segs) | Mārvā F1 0.911 (727 segs) |
| | Basant 0.531 (491) | Śrī 0.834 (1095) |
| | Yaman kalyāṇ 0.543 (752) | Bairāgi 0.830 (554) |

Dēś, the shortest raga, is the weakest class. But the relationship is far from
deterministic — Mārvā is strongest on middling data, Yaman kalyāṇ weak despite
752 segments — so duration bias is a real contributor, not the whole story. It
is a concrete argument for testing a per-recording segment cap.

### Where this leaves the project

| | Saraga | HMD full |
|---|---|---|
| ragas | 6 | **30** |
| recordings | 13 | **300** |
| best track accuracy | 0.615 | **0.9300** (0.8633 album-grouped) |
| × chance (segment) | 3.1× | **21.2×** |

**The ceiling was sample scarcity, and it is gone.** Accuracy went *up* while
the problem got five times harder. The 2026-08-16 corrections (leaky split,
wrong tonic) plus the scale-up together turn an indefensible 0.9051 on 6 ragas
into a defensible 0.93 on 30.

The LR/RF pattern completes: RF ties or edges LR on track accuracy but loses on
segment accuracy in both groupings, and its deficit widens under the stricter
one (−0.038 track-grouped, −0.038 album-grouped on segments). Consistent with
every observation since the tonic was corrected.

### Artifacts

`outputs/classifier_runs_hmd-full-30raga/` holds one JSON and one readable
report per grouping, each with per-class precision/recall/F1, the per-raga
tracks/segments/CV-groups bias table, and every misclassified recording listed.

### Next steps

- **Segment cap experiment**: cap segments per recording (e.g. at the 64-segment median) so long recordings cannot dominate, and compare. This is a real experimental variable, not cleanup, so it should be run alongside the uncapped result rather than replacing it.
- Estimated-tonic variant of this run, to confirm the tonic effect at 30 classes as it was confirmed at 5.
- `run_tonic_validation.py` across all 300 HMD tracks.
- Fix the tonic estimator's octave/fifth resolution — still the one unaddressed root cause.