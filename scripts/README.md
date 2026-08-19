# scripts/

Driver scripts that *consume* the `commentator` package. **None of these are
pytest tests** — this directory was called `tests/` until 2026-08-19, which is
why several files were once named `test_*.py` and had to be renamed to stop
pytest collecting them. The repo's only real unit test is
`commentator/tests/test_pitch_contour.py`.

Run everything from the repo root so the package imports resolve:

```bash
.venv/bin/python scripts/<name>.py
```

## Current

| script | what it does |
|---|---|
| `run_tonic_validation.py` | Measures Stage-1 tonic error against annotated ground truth, at both track and segment level. Produces HANDOFF §2d. The segment-level number is the one that matters — the classifier sees a per-window tonic. |
| `run_compmusic_check.py` | Smoke test for the CompMusic HMD adapter: 300 recordings, 30 ragas, MBID join. Run it after touching `commentator/io/compmusic.py`. `--download` fetches ~3.4 GB on first use. |
| `run_pipeline_sanity_check.py` | End-to-end check on one Saraga track: Stage-1 schema → feature vector → the old KNN baseline. Quickest way to confirm the pipeline still runs at all. |
| `inspect_features.py` | Exploratory **track-level** feature tables and t-SNE. Writes to `outputs/inspect_features/`, deliberately separate from `outputs/runs/`, which holds segment-level extractions. |
| `run_experiment_three.py` | The Gemini-based LLM commentary experiment (Stage 3). Lived in `commentator/analysis/` until 2026-08-19; it is a driver, not library code. Needs `GEMINI_API_KEY` in `.env`. Writes to `outputs/commentary/`. Experiments one and two are in the workspace-level `llm/` directory, outside this repo. |

**The classifier entry point is `run_segment_lr_rf.py` at the repo root**, not
here — it is the project's main program rather than a helper.

## legacy/

Superseded, kept for reproducibility. Nothing here should produce a reported
number.

| file | why it is here |
|---|---|
| `classifier_compare.py` | Single `GroupShuffleSplit`; the script behind the retracted 0.9051. Superseded by `run_segment_lr_rf.py --from-run`. See its module docstring. |
| `run_segment_baseline_knn.py` | The original KNN baseline over 13 hardcoded Saraga tracks. Superseded by `run_segment_lr_rf.py`. |
| `run_mirdata_check.py` | Early mirdata download/listing check for Saraga. The project now reads HMD straight off disk and Saraga through its adapter. |
| `test_raga.ipynb` | Early exploration notebook. |

Deleted 2026-08-19: `run_stage1_sanity_check.py` — it evaluated
`result["meta"]`, `result["tonic"]` etc. as bare expressions with no `print`,
so running it produced no output at all. A notebook cell saved as a file.
