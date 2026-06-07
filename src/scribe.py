import os
from groq import Groq
from src.state import MedicalBoardState

def run_scribe_node(state: MedicalBoardState) -> dict:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    all_chat_lines = "\n".join(state["raw_debate_history"])
    
    prompt = f"""
    Summarize this medical panel discussion turn into concise, integrated clinical progress tracking points.
    Previous Notes: {state['compressed_transcript']}
    New Chat log lines: {all_chat_lines}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {"compressed_transcript": completion.choices[0].message.content, "raw_debate_history": []}
