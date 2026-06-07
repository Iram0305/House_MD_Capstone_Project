import os
from groq import Groq
from src.state import MedicalBoardState

def run_cmo_node(state: MedicalBoardState) -> dict:
    print("\n--- PHASE 4: CMO SYNTHESIS REPORT ---")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    evidence = state.get("evidence_payload", {})
    verified_context = {d: p for d, p in evidence.items() if p["orphadata_match"] or p["pubmed_citations"]}
    
    prompt = f"""
    Act as the Chief Medical Officer. Compile the final definitive clinical diagnostic support document.
    Include only conditions backed by verified tokens.
    
    Verified Sources: {verified_context}
    Board Transcript: {state['compressed_transcript']}
    Format layout with structured clinical headings.
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {"final_report": {"free_text_report": completion.choices[0].message.content}}
