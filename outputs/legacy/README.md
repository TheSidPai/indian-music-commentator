# outputs/legacy/

Superseded artifacts. **Nothing here should be cited.** Kept so retracted and
pre-restructure figures stay reproducible.

## groupshufflesplit/

Nine files produced by `scripts/legacy/classifier_compare.py`, which splits with
a single `GroupShuffleSplit` rather than the project's frozen protocol
(StratifiedGroupKFold + album group map + segment→track majority vote).

They were sitting **inside three runs' `eval/` directories**, unmarked, beside
frozen-protocol results — with nothing to tell a reader they came from a weaker
split. Moved here 2026-08-19. Organised by the run whose features they used:

- `2026-08-16_saraga-6raga_annotated/`
- `2026-08-16_saraga-6raga_estimated/`
- `2026-08-17_hmd-full-30raga_annotated/`

For the frozen-protocol results on those same feature tables, see each run's own
`eval/` directory, and `outputs/INDEX.md` for the cross-run table.

## Pre-restructure feature tables and plots

`key_features_table.csv`, `key_segment_features_table.csv`, `tsne_plot.png`,
`tsne_segments.png` — generated artifacts that were committed inside the old
`tests/` directory. `key_segment_features_table.csv` is the pre-restructure name
of what is now `runs/<run_id>/features.csv.gz`. The 2026-08-19 outputs
restructure migrated the copies under `outputs/` and missed these four; moved
here 2026-08-19.

`tsne_segments_prerun-tag_duplicate.png` — a byte-identical duplicate from
before run directories existed.
