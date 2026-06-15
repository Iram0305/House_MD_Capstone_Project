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
            time.sleep(1)
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            if attempt == max_retries - 1:
                raise e
            status_code = getattr(e, 'status_code', '5xx/Timeout')
            print(f"⚠️ [Boardroom API Error - Status {status_code}]: Retrying agent activation in {delay} seconds...")
            time.sleep(delay)
            delay *= 2

def run_specialist_debate(state: MedicalBoardState, specialty_name: str, specialty_focus: str) -> dict:
    client = genai.Client()
    symptom_list = [f"{code} ({state['hpo_labels'][code]})" for code in state['validated_hpo_codes']]
    
    system_instruction = f"You are a world-class Medical Specialist in {specialty_name}. Focus: {specialty_focus}. Be concise, clear, and omit unnecessary conversational pleasantries."
    
    user_prompt = f"""
    Review these patient symptoms: {symptom_list}
    
    Current board summary notes from previous discussions:
    {state['compressed_transcript']}
    
    Provide your clinical analysis. Debate any opinions from other specialists if they seem incorrect. 
    At the very end of your response, output your updated top 3 rare disease guesses inside square brackets like this: 
    Final Guesses: [Disease A, Disease B, Disease C]
    """
    
    response = generate_with_retry(
        client=client,
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7
        )
    )
    
    argument_output = response.text
    chat_line = f"[{specialty_name} Doctor]: {argument_output}"
    print(chat_line)
    
    try:
        guess_str = argument_output.split("[")[-1].split("]")[0]
        guesses = [g.strip() for g in guess_str.split(",")]
    except Exception:
        guesses = ["Undetermined Rare Condition"]

    updated_guesses = list(state.get("current_guesses", []))
    updated_guesses.append({"specialty": specialty_name, "candidates": guesses})
    
    updated_history = list(state.get("raw_debate_history", []))
    updated_history.append(chat_line)
    
    updated_full_log = list(state.get("full_debate_log", []))
    updated_full_log.append(chat_line)
    
    return {
        "raw_debate_history": updated_history, 
        "current_guesses": updated_guesses,
        "full_debate_log": updated_full_log
    }

def neurologist_node(state: MedicalBoardState):
    return run_specialist_debate(state, "Neurology", "Brain, spinal cord, and central nervous pathways.")

def immunologist_node(state: MedicalBoardState):
    return run_specialist_debate(state, "Clinical Immunology", "Systemic immune disorders and autoimmune reactions.")

def geneticist_node(state: MedicalBoardState):
    return run_specialist_debate(state, "Medical Genetics", "Congenital DNA syndromes and inherited metabolic blockages.")
