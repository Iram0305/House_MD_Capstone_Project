import os
from google import genai
from google.genai import types
from src.state import MedicalBoardState

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
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0
        )
    )
    return {"final_report": {"free_text_report": response.text}}
