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
                print(f"⚠️ [Scribe Rate Limit]: Retrying transcript compression in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e

def run_scribe_node(state: MedicalBoardState) -> dict:
    client = genai.Client()
    all_chat_lines = "\n".join(state["raw_debate_history"])
    
    prompt = f"""
    Summarize this medical panel discussion turn into concise, integrated clinical progress tracking points.
    Previous Notes: {state['compressed_transcript']}
    New Chat log lines: {all_chat_lines}
    """
    
    response = generate_with_retry(
        client=client,
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0)
    )
    return {"compressed_transcript": response.text, "raw_debate_history": []}
