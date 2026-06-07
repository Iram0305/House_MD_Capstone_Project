import os
from groq import Groq
from src.state import MedicalBoardState

def run_specialist_debate(state: MedicalBoardState, specialty_name: str, specialty_focus: str) -> dict:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    symptom_list = [f"{code} ({state['hpo_labels'][code]})" for code in state['validated_hpo_codes']]
    
    prompt = f"""
    You are a world-class Medical Specialist in {specialty_name}. Core Target: {specialty_focus}.
    Review these patient symptoms: {symptom_list}
    
    Current board summary notes:
    {state['compressed_transcript']}
    
    Provide your clinical analysis. At the very end of your response, output your updated top 3 rare disease guesses inside square brackets like this: Final Guesses: [Disease A, Disease B, Disease C]
    """
    
    # Utilizing Groq's high-capacity reasoning model
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    argument_output = completion.choices[0].message.content
    chat_line = f"[{specialty_name} Doctor]: {argument_output}"
    
    try:
        guess_str = argument_output.split("[")[-1].split("]")[0]
        guesses = [g.strip() for g in guess_str.split(",")]
    except Exception:
        guesses = ["Undetermined Rare Condition"]

    updated_guesses = list(state.get("current_guesses", []))
    updated_guesses.append({"specialty": specialty_name, "candidates": guesses})
    
    updated_history = list(state.get("raw_debate_history", []))
    updated_history.append(chat_line)
    
    return {"raw_debate_history": updated_history, "current_guesses": updated_guesses}

def neurologist_node(state: MedicalBoardState):
    return run_specialist_debate(state, "Neurology", "Brain and central nervous pathways.")

def immunologist_node(state: MedicalBoardState):
    return run_specialist_debate(state, "Clinical Immunology", "Systemic immune disorders.")

def geneticist_node(state: MedicalBoardState):
    return run_specialist_debate(state, "Medical Genetics", "Congenital DNA syndromes.")
