import os
from groq import Groq
from src.state import MedicalBoardState

def run_specialist_debate(state: MedicalBoardState, specialty_name: str, specialty_focus: str) -> dict:
    # Authenticate safely using your remote cloud dashboard keys
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # Format the validated HPO symptom codes cleanly for the Llama context window
    symptom_list = [f"{code} ({state['hpo_labels'][code]})" for code in state['validated_hpo_codes']]
    
    prompt = f"""
    You are a world-class Medical Specialist in {specialty_name}. Core Target Focus: {specialty_focus}.
    Review these patient symptoms: {symptom_list}
    
    Current board summary notes from previous discussions:
    {state['compressed_transcript']}
    
    Provide your clinical analysis. Debate any opinions from other specialists if they seem incorrect. 
    At the very end of your response, output your updated top 3 rare disease guesses inside square brackets like this: 
    Final Guesses: [Disease A, Disease B, Disease C]
    """
    
    # Call the high-speed Llama inference engine on Groq Cloud
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    argument_output = completion.choices[0].message.content
    chat_line = f"[{specialty_name} Doctor]: {argument_output}"
    print(chat_line)
    
    # Simple isolation logic to strip out candidate strings safely
    try:
        guess_str = argument_output.split("[")[-1].split("]")[0]
        guesses = [g.strip() for g in guess_str.split(",")]
    except Exception:
        guesses = ["Undetermined Rare Condition"]

    # Save details to short-term list containers for processing
    updated_guesses = list(state.get("current_guesses", []))
    updated_guesses.append({"specialty": specialty_name, "candidates": guesses})
    
    updated_history = list(state.get("raw_debate_history", []))
    updated_history.append(chat_line)
    
    # Commit a permanent copy to our UI history tracker log
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
