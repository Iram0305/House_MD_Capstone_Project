from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.state import MedicalBoardState


# The schema forces the AI doctors to supply a list of disease guesses alongside their text arguments
class DoctorOutput(BaseModel):
    argument: str = Field(description="Your clinical assessment and counter-arguments against other doctors.")
    diagnoses: list[str] = Field(description="Your updated top 3 rare disease differential guesses.")


def run_specialist_debate(state: MedicalBoardState, specialty_name: str, specialty_focus: str) -> dict:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)  # Slightly higher temperature allows active debate
    structured_llm = llm.with_structured_output(DoctorOutput)

    # Collect the symptoms and their official readable labels
    symptom_list = [f"{code} ({state['hpo_labels'][code]})" for code in state['validated_hpo_codes']]

    # Build the workspace prompt for the doctor
    system_instruction = f"""You are a world-class Medical Specialist in {specialty_name}. 
    Your core focus is: {specialty_focus}.
    Review the standardized symptom codes and the ongoing medical board discussion summary.
    Provide your unique clinical assessment. If another specialist's guess is weak, politely explain why."""

    user_prompt = f"""
    Official Patient Symptoms: {symptom_list}

    Medical Board Running Summary Notes:
    {state['compressed_transcript']}

    Recent Chat Lines this Turn:
    {state['raw_debate_history']}

    Analyze the case data, debate other opinions, and output your clinical analysis text alongside your top 3 rare disease guesses.
    """

    response = structured_llm.invoke([
        ("system", system_instruction),
        ("human", user_prompt)
    ])

    # Format the speaker tag line
    chat_line = f"[{specialty_name} Doctor]: {response.argument} | My Top Guesses: {response.diagnoses}"
    print(f"\n{chat_line}")

    # Copy existing guesses and update them
    updated_guesses = list(state.get("current_guesses", []))
    updated_guesses.append({"specialty": specialty_name, "candidates": response.diagnoses})

    # Add this message line to the round's raw chat list
    updated_history = list(state.get("raw_debate_history", []))
    updated_history.append(chat_line)

    return {
        "raw_debate_history": updated_history,
        "current_guesses": updated_guesses
    }


# Node entrypoints for our Graph structure
def neurologist_node(state: MedicalBoardState):
    return run_specialist_debate(state, "Neurology",
                                 "Brain, spinal cord, central nervous system anomalies, and nerve pathways.")


def immunologist_node(state: MedicalBoardState):
    return run_specialist_debate(state, "Clinical Immunology",
                                 "Autoimmune reactions, systemic hyper-inflammation, and immune dysfunction.")


def geneticist_node(state: MedicalBoardState):
    return run_specialist_debate(state, "Medical Genetics",
                                 "Congenital syndromic presentations, inherited metabolic blockages, and rare gene disruptions.")