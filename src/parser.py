import json
import os
from groq import Groq
from src.state import MedicalBoardState

def run_parser_node(state: MedicalBoardState) -> dict:
    print("\n--- PHASE 1: LIGHTWEIGHT GROQ PARSER ---")
    
    # Connects to Groq using the cloud key
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    prompt = f"""
    Extract all physical symptoms or clinical terms from this patient story. 
    Return them as a simple comma-separated string list (e.g., tremor, fatigue, weakness).
    Do not add introductory chat filler.
    
    Patient Story: {state['raw_narrative']}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    extracted_phrases = [p.strip().lower() for p in completion.choices[0].message.content.split(",")]
    print(f"Groq Extracted Items: {extracted_phrases}")
    
    validated_codes = []
    labels_map = {}
    
    # Read the text file directly into background memory safely
    if os.path.exists("data/hp-base.json"):
        with open("data/hp-base.json", "r") as f:
            hpo_data = json.load(f)
            
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
