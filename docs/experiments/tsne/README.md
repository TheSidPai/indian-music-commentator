# t-SNE plot archive

`plot_tsne_segments(...)` (segment-level) and `plot_tsne(...)` in
`scripts/inspect_features.py` (track-level) always saved to a fixed filename,
so each re-run with a different raga subset / feature set / segment window
silently overwrote the previous image. This folder backfills the ones that
were lost, and consolidates the ones that happened to survive by having been
committed at the right moment. See `docs/experiments/2026-06-raga-baseline-log.md`
for the full narrative behind each run.

| File | Date | Level | Ragas | Segment window | Features | Provenance |
|---|---|---|---|---|---|---|
| `2026-06-15_track-level_3raga-bihag-yaman-kedar_19feat.png` | 2026-06-15 | Track | Bihag, Yaman, Kedar | n/a (whole track) | 19 (hand-picked subset, see `scripts/inspect_features.py`) | Original file, copied from `outputs/legacy/tsne_plot.png` — only ever generated once |
| `2026-06-16_segment_3raga-bihag-bhoopali-kedar_52feat_30s-20shop.png` | 2026-06-16 (v1b) | Segment | Bihag, Bhoopali, Kedar | 30s window / 20s hop | 52 (pre-subset, pre-contour) | **Regenerated** using commit `ee331c5`'s code via a disposable `git worktree`. Segment count (910) matches `docs/experiments/segment_3_raag_result.txt` exactly. Note: the log's prose for this entry says "608 valid segments" and "same settings" as the non-overlapping 30s/30s v1 run, but the actual saved terminal output (`segment_3_raag_result.txt`) shows 910 segments — only reproducible with a 30s/20s hop. This file matches the verified raw artifact, not the prose summary. |
| `2026-06-18_segment_6raga_52feat_30s-20shop.png` | 2026-06-18 | Segment | All 6 (+ Abhogi, Shree, Lalit) | 30s window / 20s hop | 52 (pre-subset, pre-contour) | **Regenerated** using commit `ee331c5`'s code. Segment count (1809) matches `docs/experiments/segment_6_raag_result.txt` exactly. |
| `2026-06-19_segment_6raga_41feat-subset_30s-20shop.png` | 2026-06-19 | Segment | Same 6 | 30s window / 20s hop | 41 (post `DROP_NAMES` subset, pre-contour) | Original file, copied from `tests/tsne_segments.png` |
| `2026-06-23_segment_6raga_71feat-contour_30s-20shop.png` | 2026-06-23 | Segment | Same 6 | 30s window / 20s hop | 71 (post-subset + contour/stability/pattern features) | Original file, copied from the repo-root `tsne_segments.png` (current pipeline state) |
| `2026-06-23_segment_6raga_71feat-contour_60s-50shop.png` | 2026-06-23 (longer-segment variant) | Segment | Same 6 | 60s window / 50s hop | 71 (same feature set as above) | **Regenerated** using today's code with `segment_length_s=60, hop_s=50`. Per the log, results were "effectively unchanged" vs. the 30s/20s run at this feature set — this plot is expected to look similar to the row above. |

Feature counts were double-checked against `X.shape` printed during each
regeneration run, not just inferred from column counts in the paired CSVs.
