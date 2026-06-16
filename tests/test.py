from commentator.io.loaders import initialize_saraga, get_pitch_for_track
from commentator.analysis.stage1_schema import build_stage1_schema

saraga = initialize_saraga("/home/thesidpai/mir_projects/data")
pitch_obj = get_pitch_for_track("27_Raag_Bihag", saraga)

result = build_stage1_schema(
    pitch_obj,
    raga_label="Bihag",
    include_artifacts=False,
)

result["meta"]
result["tonic"]
result["swara"]
result["pitch_summary"]
result["comments"]