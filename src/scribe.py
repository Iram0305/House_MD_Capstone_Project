import os
from google import genai
from google.genai import types
from src.state import MedicalBoardState

def run_scribe_node(state: MedicalBoardState) -> dict:
    client = genai.Client()
    all_chat_lines = "\n".join(state["raw_debate_history"])
    
    prompt = f"""
    Summarize this medical panel discussion turn into concise, integrated clinical progress tracking points.
    Previous Notes: {state['compressed_transcript']}
    New Chat log lines: {all_chat_lines}
    """
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0)
    )
    return {"compressed_transcript": response.text, "raw_debate_history": []}
