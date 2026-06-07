from langchain_openai import ChatOpenAI
from src.state import MedicalBoardState


def run_scribe_node(state: MedicalBoardState) -> dict:
    print("\n--- SCRIBE AGENT: COMPRESSING TRANSCRIPT ---")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    all_chat_lines = "\n".join(state["raw_debate_history"])

    prompt = f"""
    You are the Medical Scribe. Read the previous running summary notes and the new debate log entries from this round.
    Synthesize them into an updated, dense, clinical progress report tracking:
    1. Hypotheses maintained or advanced
    2. Differentials rejected and why
    3. Overlapping symptoms agreed upon

    Previous Notes:
    {state['compressed_transcript']}

    New Debate Entries:
    {all_chat_lines}

    Output the updated clean clinical summary. Do not include introductory conversational text.
    """

    summary = llm.invoke(prompt).content

    # Clear the raw chat history for this turn and pass down the compressed notes
    return {
        "compressed_transcript": summary,
        "raw_debate_history": []  # Resetting for next turn if loop continues
    }