import os
import time
import groq
from groq import Groq
from src.state import MedicalBoardState

def call_groq_with_backoff(client, model, messages, temperature=0):
    delay = 4  
    max_retries = 5
    for attempt in range(max_retries):
        try:
            time.sleep(1)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return completion
        except groq.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"⚠️ [Scribe Rate Limit Hit]: Retrying summary aggregation in {delay}s...")
            time.sleep(delay)
            delay *= 2

def run_scribe_node(state: MedicalBoardState) -> dict:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    all_chat_lines = "\n".join(state["raw_debate_history"])
    
    prompt = f"""
    Summarize this medical panel discussion turn into concise, integrated clinical progress tracking points.
    Previous Notes: {state['compressed_transcript']}
    New Chat log lines: {all_chat_lines}
    """
    
    completion = call_groq_with_backoff(
        client=client,
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {"compressed_transcript": completion.choices[0].message.content, "raw_debate_history": []}
