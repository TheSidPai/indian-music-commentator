import os
import time
import csv
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import mirdata

# Absolute imports matching your package module execution context
from commentator.core.pitch_contour import PitchContour
from commentator.analysis.stage1_schema import build_stage1_schema
from commentator.analysis.swara_analyzer import generate_basic_swara_comment
from commentator.io.saraga import load_track

# Repo root, resolved from this file's location (scripts/ is one level down),
# so .env and outputs/ are found regardless of the caller's cwd.
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

try:
    from google import genai
    client = genai.Client()
except ImportError:
    client = None

# ==============================================================================
# PROMPT REGISTRY
# ==============================================================================
PROMPT_HYBRID_V1 = """
You are an expert Hindustani Classical Music Commentator. 
You are listening to a raw performance audio stream, and you have been provided with highly accurate, aggregated Stage-1 MIR statistical features extracted from the track's pitch contour.

Verified Stage-1 MIR Features:
- Estimated Performance Tonic (Sa): {tonic_hz} Hz
- Confidently Detected Dominant Swaras: {dominant_swaras}
- Least Used / Omitted Swaras: {least_used_swaras}
- Voiced Pitch Cent Boundaries: Min: {min_cents:.2f} cents, Max: {max_cents:.2f} cents
- Intermediate Musicological Summary: {basic_comment}

Your Task:
Write a comprehensive, detailed, and fluent musical commentary tailored for a target user who is an '{role}'. 

Synthesize what you hear dynamically in the raw audio with the verified statistical framework provided above. Comment on the progression of the performance structure (Alap, Jor, Jhala, or Bandish variations), the character of the voice or instrument, register focus, and ornamentation (meend, taan, or gamak). Your raga analysis must strictly align with the dominant swaras and interval ranges extracted by the pipeline. Do not introduce notes that are listed as omitted.

Speak authoritatively. Do not state you are an AI or explicitly mention you are reading a text prompt framework.
"""

ACTIVE_PROMPT_TEMPLATE = PROMPT_HYBRID_V1


def get_track_audio_path(track_obj) -> Path | None:
    """
    Locates the audio file path, falling back to the workspace audio_folder
    if mirdata cannot find it in its central cache.
    """
    print(track_obj.audio_path)
    if hasattr(track_obj, 'audio_path') and track_obj.audio_path:
        return track_obj.audio_path
            
    # Absolute path fallback matching directory configuration. This is the
    # workspace-level audio_folder (sibling of the repo), not one inside it.
    workspace_audio_dir = ROOT_DIR.parent / "llm" / "audio_folder"
    track_id = track_obj.track_id
    
    # Clean track numeric prefixes if present (e.g., '20_Raag_Abhogi' -> 'Raag_Abhogi.mp3')
    clean_id = track_id.split('_', 1)[-1] if '_' in track_id and track_id.split('_')[0].isdigit() else track_id
    
    possible_names = [f"{track_id}.mp3", f"{track_id}.wav", f"{clean_id}.mp3", f"{clean_id}.wav"]
    for name in possible_names:
        fallback_path = workspace_audio_dir / name
        if fallback_path.exists():
            return fallback_path
            
    return None


def extract_hybrid_text_features(pitch_contour_obj: PitchContour, raga_label: str | None = None) -> dict:
    """
    Runs your Stage-1 core schema pipeline using the verified PitchContour object.
    """
    # Build your schema matching structural specifications
    stage1 = build_stage1_schema(pitch_contour_obj, raga_label=raga_label, include_artifacts=True)
    
    tonic_block = stage1.get("tonic", {})
    swara_block = stage1.get("swara", {})
    pitch_summary = stage1.get("pitch_summary", {})
    comments = stage1.get("comments", {})

    artifacts = stage1.get("artifacts", {})
    swara_assignment = artifacts.get("swara_assignment", {})

    dominant_swaras = swara_block.get("dominant_swaras") or swara_assignment.get("assignedswaras", []) or []
    
    range_summary = pitch_summary.get("voiced_range_cents", {})
    min_cents = range_summary.get("min_cents") or 0.0
    max_cents = range_summary.get("max_cents") or 0.0
    
    meta = stage1.get("meta", {})
    tonic_hz = tonic_block.get("tonic_hz") or 0.0
    
    mock_tonic_result = {
        "track_id": meta.get("track_id", "unknown"),
        "tonic_hz": tonic_hz
    }
    
    swara_comment_source = swara_block if swara_block else swara_assignment
    basic_comment = comments.get("basic_comment") or generate_basic_swara_comment(mock_tonic_result, swara_comment_source)
    
    all_canonical_swaras = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma^", "Pa", "dha", "Dha", "ni", "Ni"]
    least_used_swaras = [s for s in all_canonical_swaras if s not in dominant_swaras]

    return {
        "tonic_hz": tonic_hz,
        "dominant_swaras": ", ".join(dominant_swaras) if dominant_swaras else "None Detected",
        "least_used_swaras": ", ".join(least_used_swaras),
        "min_cents": min_cents,
        "max_cents": max_cents,
        "basic_comment": basic_comment
    }


def main():
    if client is None:
        print("Error: google-genai package is missing.")
        return

    outputs_dir = ROOT_DIR / "outputs" / "commentary"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_csv = outputs_dir / "experiment_three_hybrid_results.csv"
    target_role = "advanced learner"
    model_name = "gemini-2.5-flash"
    
    print("Initializing Saraga Hindustani dataset tracker...")
    data_home = os.environ.get("DATA_HOME")
    saraga = mirdata.initialize("saraga_hindustani", data_home=data_home)
    
    # Target subset operational array
    test_track_ids = [tid for tid in saraga.track_ids if any(r in tid for r in ["Bihag"])][:4]
    if not test_track_ids:
        test_track_ids = saraga.track_ids[:4]

    write_header = not output_csv.exists()
    csv_file = open(output_csv, 'a', encoding='utf-8', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=[
        'track_id', 'true_raga', 'audio_path', 'model_id', 'prompt_version',
        'role', 'tonic_hz', 'dominant_swaras', 'llm_output_text', 'status', 'timestamp_unix'
    ])
    if write_header:
        writer.writeheader()

    for track_id in test_track_ids:
        print(f"\nProcessing Hybrid Alignment Pipeline for Track: {track_id}")
        
        track_data = saraga.track(track_id)
        audio_path = get_track_audio_path(track_data)
        
        if not audio_path:
            print(f"Skipping track {track_id}: Physical media file missing from directory context.")
            continue
            
        true_raga = getattr(track_data, 'raga', {}).get('name', 'Unknown')
        started = int(time.time())
        status = 'ok'
        output_text = ''
        
        try:
            print("Loading track pitch contour through native io utils...")
            # The loader handles from_f0data wrapping automatically under the hood
            pitch_contour_obj = load_track(track_id, saraga)
            
            print("Extracting high-level symbolic text summaries from contours...")
            features = extract_hybrid_text_features(pitch_contour_obj, raga_label=true_raga)
            
            prompt = ACTIVE_PROMPT_TEMPLATE.format(
                role=target_role,
                tonic_hz=features['tonic_hz'],
                dominant_swaras=features['dominant_swaras'],
                least_used_swaras=features['least_used_swaras'],
                min_cents=features['min_cents'],
                max_cents=features['max_cents'],
                basic_comment=features['basic_comment']
            )
            
            print("Uploading asset payload to cloud API instance...")
            audio_blob = client.files.upload(file=Path(audio_path))
            
            print("Generating grounded, expert hybrid commentary...")
            response = client.models.generate_content(
                model=model_name,
                contents=[audio_blob, prompt]
            )
            
            output_text = response.text if response.text else "ERROR: Received empty string."
            client.files.delete(name=audio_blob.name) # type: ignore
            
        except Exception as e:
            status = f"error: {type(e).__name__}"
            output_text = f"ERROR: Execution failed: {str(e)}"
            print(f"Pipeline crashed for track {track_id}: {output_text}")

        row = {
            'track_id': track_id,
            'true_raga': true_raga,
            'audio_path': str(audio_path),
            'model_id': model_name,
            'prompt_version': 'exp3_v1_hybrid_grounded',
            'role': target_role,
            'tonic_hz': features.get('tonic_hz', 0.0) if 'features' in locals() else 0.0, # type: ignore
            'dominant_swaras': features.get('dominant_swaras', '') if 'features' in locals() else '', # type: ignore
            'llm_output_text': output_text.strip(),
            'status': status,
            'timestamp_unix': started
        }
        writer.writerow(row)
        csv_file.flush()
        print(f"[{status}] Transaction logging completed for {track_id}")
        
        time.sleep(5)

    csv_file.close()
    print(f"\nExperiment Three Run Completed. Dataset logs saved to: {output_csv}")


if __name__ == '__main__':
    main()