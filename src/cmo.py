import os
import time
from google import genai
from google.genai import types
from google.genai import errors
from src.state import MedicalBoardState

def generate_with_retry(client, model, contents, config=None):
    delay = 4  
    max_retries = 5
    for attempt in range(max_retries):
        try:
            time.sleep(1)
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.ClientError as e:
            if getattr(e, 'status_code', None) == 429 or "429" in str(e):
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️ [CMO Rate Limit]: Retrying executive synthesis in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e

def run_cmo_node(state: MedicalBoardState) -> dict:
    print("\n--- PHASE 4: CMO SYNTHESIS REPORT ---")
    client = genai.Client()
    
    evidence = state.get("evidence_payload", {})
    verified_context = {d: p for d, p in evidence.items() if p["orphadata_match"] or p["pubmed_citations"]}
    
    system_instruction = "Act as the Chief Medical Officer."
    
    user_prompt = f"""
    Compile the final definitive clinical diagnostic support document.
    Include only conditions backed by verified tokens.
    
    Verified Sources: {verified_context}
    Board Transcript: {state['compressed_transcript']}
    Format layout with structured clinical headings.
    """
    
    response = generate_with_retry(
        client=client,
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0
        )
    )
    return {"final_report": {"free_text_report": response.text}}
