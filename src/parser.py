import json
import os
import time
from google import genai
from google.genai import types
from google.genai import errors
from src.state import MedicalBoardState

def generate_with_retry(client, model, contents, config=None):
    """Safely handles Gemini API calls with automatic retry backoff for all API errors."""
    delay = 4  
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Baseline pacing to prevent hitting rapid burst traffic filters
            time.sleep(1)
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            # Catches ClientError (429) AND ServerError (500/503)
            if attempt == max_retries - 1:
                raise e
            status_code = getattr(e, 'status_code', '5xx/Timeout')
            print(f"⚠️ [Gemini API Error - Status {status_code}]: Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2

def run_parser_node(state: MedicalBoardState) -> dict:
    print("\n--- PHASE 1: LIGHTWEIGHT GEMINI PARSER ---")
    client = genai.Client()
    
    prompt = f"""
    Extract all physical symptoms or clinical terms from this patient story. 
    Return them as a simple comma-separated string list (e.g., tremor, fatigue, weakness).
    Do not add introductory chat filler.
    
    Patient Story: {state['raw_narrative']}
    """
    
    response = generate_with_retry(
        client=client,
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0)
    )
    
    extracted_phrases = [p.strip().lower() for p in response.text.split(",")]
    print(f"Gemini Extracted Items: {extracted_phrases}")
    
    validated_codes = []
    labels_map = {}
    
    if os.path.exists("data/hp-base.json"):
        with open("data/hp-base.json", "r") as f:
            hpo_data = json.load(f)
            
        # FIXED: Removed the trailing colon from the line below
        nodes = hpo_data.get("graphs", [{}])[0].get("nodes", [])
        
        for phrase in extracted_phrases:
            if not phrase: continue
            for node in nodes:
                if "id" in node and "lbl" in node and "HP_" in node["id"]:
                    label = node["lbl"].lower()
                    if phrase in label or label in phrase:
                        hpo_id = node["id"].split("/")[-1].replace("_", ":")
                        if hpo_id not in validated_codes:
                            validated_codes.append(hpo_id)
                            labels_map[hpo_id] = node["lbl"]
                            break 
                            
    print(f"Validated HPO Codes: {validated_codes}")
    return {
        "validated_hpo_codes": validated_codes,
        "hpo_labels": labels_map,
        "debate_turn_counter": 1
    }
