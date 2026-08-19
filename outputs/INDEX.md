# Experiment index

One row per evaluation. Appended automatically by `run_segment_lr_rf.py`.
Feature tables, plots and manifests live in `outputs/runs/<run>/`;
every evaluation of a run's features is in that run's `eval/`.

| run | dataset | ragas | tracks | feats | tonic | grouping | CV | model | track | segment | xchance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-16_saraga-6raga_annotated | saraga | 6 | 13 | 71 | ann | track | LeaveOneGroupOut | Logistic Regression | 0.6150 | 0.5200 | 3.1x |
| 2026-08-16_saraga-6raga_annotated | saraga | 6 | 13 | 71 | ann | track | LeaveOneGroupOut | Random Forest | 0.5380 | 0.4410 | 2.6x |
| 2026-08-16_saraga-6raga_estimated | saraga | 6 | 13 | 71 | est | track | LeaveOneGroupOut | Logistic Regression | 0.6150 | 0.3750 | 2.2x |
| 2026-08-16_saraga-6raga_estimated | saraga | 6 | 13 | 71 | est | track | LeaveOneGroupOut | Random Forest | 0.6150 | 0.3840 | 2.3x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 71 | ann | track | StratifiedGroupKFold(10) | Logistic Regression | 0.9300 | 0.7079 | 21.2x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 71 | ann | track | StratifiedGroupKFold(10) | Random Forest | 0.9333 | 0.6735 | 20.2x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 71 | ann | album | StratifiedGroupKFold(4) | Logistic Regression | 0.8633 | 0.6648 | 19.9x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 71 | ann | album | StratifiedGroupKFold(4) | Random Forest | 0.8533 | 0.6264 | 18.8x |
| 2026-08-17_hmd-pilot1-30raga-1each_estimated | compmusic_hmd | 30 | 30 | 71 | est | - | - | *(no classifier run)* | - | - | - |
| 2026-08-17_hmd-pilot2-5raga-10each_annotated | compmusic_hmd | 5 | 50 | 71 | ann | track | StratifiedGroupKFold(10) | Logistic Regression | 1.0000 | 0.9391 | 4.7x |
| 2026-08-17_hmd-pilot2-5raga-10each_annotated | compmusic_hmd | 5 | 50 | 71 | ann | track | StratifiedGroupKFold(10) | Random Forest | 1.0000 | 0.9303 | 4.7x |
| 2026-08-17_hmd-pilot2-5raga-10each_annotated | compmusic_hmd | 5 | 50 | 71 | ann | album | StratifiedGroupKFold(5) | Logistic Regression | 0.9600 | 0.9193 | 4.6x |
| 2026-08-17_hmd-pilot2-5raga-10each_annotated | compmusic_hmd | 5 | 50 | 71 | ann | album | StratifiedGroupKFold(5) | Random Forest | 0.9200 | 0.8882 | 4.4x |
| 2026-08-17_hmd-pilot2-5raga-10each_estimated | compmusic_hmd | 5 | 50 | 71 | est | track | StratifiedGroupKFold(10) | Logistic Regression | 0.9400 | 0.6673 | 3.3x |
| 2026-08-17_hmd-pilot2-5raga-10each_estimated | compmusic_hmd | 5 | 50 | 71 | est | track | StratifiedGroupKFold(10) | Random Forest | 1.0000 | 0.8160 | 4.1x |
| 2026-08-17_hmd-pilot2-5raga-10each_estimated | compmusic_hmd | 5 | 50 | 71 | est | album | StratifiedGroupKFold(5) | Logistic Regression | 0.8600 | 0.6502 | 3.3x |
| 2026-08-17_hmd-pilot2-5raga-10each_estimated | compmusic_hmd | 5 | 50 | 71 | est | album | StratifiedGroupKFold(5) | Random Forest | 0.9600 | 0.7981 | 4.0x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 71 (control) | ann | track | StratifiedGroupKFold(10) | Logistic Regression | 0.9300 | 0.7079 | 21.2x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 71 (control) | ann | track | StratifiedGroupKFold(10) | Random Forest | 0.9333 | 0.6735 | 20.2x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 71 (control) | ann | album | StratifiedGroupKFold(4) | Logistic Regression | 0.8633 | 0.6648 | 19.9x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 71 (control) | ann | album | StratifiedGroupKFold(4) | Random Forest | 0.8533 | 0.6264 | 18.8x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 62 (passA) | ann | track | StratifiedGroupKFold(10) | Logistic Regression | 0.9500 | 0.7224 | 21.7x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 62 (passA) | ann | track | StratifiedGroupKFold(10) | Random Forest | 0.9233 | 0.6718 | 20.2x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 62 (passA) | ann | album | StratifiedGroupKFold(4) | Logistic Regression | 0.9267 | 0.7006 | 21.0x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 62 (passA) | ann | album | StratifiedGroupKFold(4) | Random Forest | 0.9267 | 0.6669 | 20.0x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 55 (passB) | ann | track | StratifiedGroupKFold(10) | Logistic Regression | 0.9500 | 0.7236 | 21.7x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 55 (passB) | ann | track | StratifiedGroupKFold(10) | Random Forest | 0.9300 | 0.6704 | 20.1x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 55 (passB) | ann | album | StratifiedGroupKFold(4) | Logistic Regression | 0.9233 | 0.7000 | 21.0x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 55 (passB) | ann | album | StratifiedGroupKFold(4) | Random Forest | 0.9367 | 0.6657 | 20.0x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 36 (passC) | ann | track | StratifiedGroupKFold(10) | Logistic Regression | 0.9567 | 0.7288 | 21.9x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 36 (passC) | ann | track | StratifiedGroupKFold(10) | Random Forest | 0.9333 | 0.6664 | 20.0x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 36 (passC) | ann | album | StratifiedGroupKFold(4) | Logistic Regression | 0.9300 | 0.7179 | 21.5x |
| 2026-08-17_hmd-full-30raga_annotated | compmusic_hmd | 30 | 300 | 36 (passC) | ann | album | StratifiedGroupKFold(4) | Random Forest | 0.9300 | 0.6631 | 19.9x |
