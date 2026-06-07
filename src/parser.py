import json
import os
import time
import groq
from groq import Groq
from src.state import MedicalBoardState

def call_groq_with_backoff(client, model, messages, temperature=0):
    """Safely executes a Groq API call with an exponential backoff sleep loop."""
    delay = 4  # Starting delay in seconds
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # Enforce a brief baseline cooling period between back-to-back nodes
            time.sleep(1)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return completion
        except groq.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e  # Propagate the error if we ran out of attempts
            print(f"⚠️ [Parser Rate Limit Hit]: Retrying execution vector in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Double the backoff duration

def run_parser_node(state: MedicalBoardState) -> dict:
    print("\n--- PHASE 1: LIGHTWEIGHT GROQ PARSER ---")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    prompt = f"""
    Extract all physical symptoms or clinical terms from this patient story. 
    Return them as a simple comma-separated string list (e.g., tremor, fatigue, weakness).
    Do not add introductory chat filler.
    
    Patient Story: {state['raw_narrative']}
    """
    
    # Execute through our protected backoff channel
    completion = call_groq_with_backoff(
        client=client,
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    extracted_phrases = [p.strip().lower() for p in completion.choices[0].message.content.split(",")]
    print(f"Groq Extracted Items: {extracted_phrases}")
    
    validated_codes = []
    labels_map = {}
    
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
