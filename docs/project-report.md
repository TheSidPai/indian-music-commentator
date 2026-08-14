# Pitch-based Raga Characterisation and Baseline Classification (Working Title)

## 1. Introduction

- Motivation: computational analysis of Hindustani ragas.
- Problem: can tonic-normalised pitch and swara statistics separate ragas?
- Scope: Saraga Hindustani, small subset of ragas, interpretable baselines.

## 2. Datasets and Subset Selection

### 2.1 Saraga Hindustani Overview
- Brief description of the dataset (source, annotations).

### 2.2 Experimental Subsets
- Table of initial and current raga subsets:
  - Bihag–Yaman–Kedar (with track IDs, durations).
  - Bihag–Bhoopali–Kedar.
- Rationale for replacing Yaman with Bhoopali (concert lengths, segment coverage).

## 3. Methodology

### 3.1 Stage‑1: Pitch and Swara Schema
- Describe pitch contour extraction, tonic estimation, swara mapping, histograms.
- Mention API choices (voicing vs confidence).

### 3.2 Stage‑2: Feature Vector
- List feature groups (tonic, swara proportions, histogram peaks, entropy, counts).
- Note that this yields a 52-dimensional feature vector per unit (track/segment).

### 3.3 Baseline Models

#### 3.3.1 Track‑level KNN
- One vector per track.
- Leave-one-track-out evaluation.

#### 3.3.2 Segment‑level KNN
- 30 s segments, hop = 30 s (and later variants).
- Majority vote to obtain track label.
- t-SNE visualisation of segment features.

## 4. Experiments and Results

### 4.1 Track‑level Baseline Results
- Accuracy, confusion patterns for Bihag–Yaman–Kedar.
- Brief discussion.

### 4.2 Segment‑level Baseline Results
- Results for Bihag–Yaman–Kedar (v1).
- Results for Bihag–Bhoopali–Kedar (v1b).
- t-SNE plots (reference to files).

## 5. Discussion

- Interpretation of:
  - Strong Bihag/Bhoopali clusters.
  - Kedar’s “bridge” behaviour.
  - Within-raga variation (especially Bihag’s two tracks).

## 6. Limitations and Future Work

- Small subset size.
- Dependence on Saraga’s pitch/voicing quality.
- Plans:
  - Overlapping segments.
  - Expanded raga set.
  - More expressive models after baselines are understood.

## 7. Conclusion

- Short summary of what is learned from this baseline phase.
