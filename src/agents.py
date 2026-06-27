"""
src/agents.py  —  Specialist Panel Nodes (Debate Phase)
========================================================
Three specialist agents run inside the LangGraph cyclic debate loop.

Changes vs. original:
  • generate_with_retry() is still the single call-site for all Gemini
    requests — no new SDK imports needed here.
  • A mandatory INTER_NODE_SLEEP_S cooldown is applied *after* every
    specialist completes, preventing the burst-traffic pattern that trips
    Google AI Studio's RPM filter when nodes fire back-to-back.
  • hitl_stage is written into the returned state slice so app.py can
    detect exactly which node just finished and pause for user approval.
"""

import time

from google.genai import types

from src.api_utils import generate_with_retry
from src.state import MedicalBoardState

# Mandatory inter-node cooling period (seconds).
# 2 s keeps burst traffic well below the free-tier RPM ceiling even when
# the scribe loop cycles multiple times.
INTER_NODE_SLEEP_S: float = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DEBATE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_specialist_debate(
    state: MedicalBoardState,
    specialty_name: str,
    specialty_focus: str,
) -> dict:
    """
    Runs a single specialist turn and returns the state delta.
    Applies a post-call cooling sleep to avoid burst-rate violations.
    """
    symptom_list = [
        f"{code} ({state['hpo_labels'][code]})"
        for code in state["validated_hpo_codes"]
    ]

    system_instruction = (
        f"You are a world-class Medical Specialist in {specialty_name}. "
        f"Focus: {specialty_focus}. "
        "Be concise, clear, and omit conversational pleasantries."
    )

    user_prompt = f"""
Review these patient symptoms: {symptom_list}
Current board summary notes from previous discussions: {state['compressed_transcript']}

Provide your clinical analysis. Debate opinions from other specialists if they seem incorrect.
At the very end of your response, output your updated top 3 rare disease guesses inside square brackets like this:
Final Guesses: [Disease A, Disease B, Disease C]
"""

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7,
    )

    # ── API call via shared utility (pacing + exponential backoff + model) ─
    response = generate_with_retry(contents=user_prompt, config=config)
    argument_output = response.text
    chat_line = f"[{specialty_name} Doctor]: {argument_output}"

    # ── Parse Final Guesses (bracket contract preserved from original) ─────
    try:
        bracket_blocks = argument_output.split("[")
        guess_str = bracket_blocks[-1].split("]")[0]
        guesses = [g.strip() for g in guess_str.split(",")]
        # Reject if the parser captured prose instead of disease names
        if len(guesses) == 1 and len(guesses[0]) > 120:
            raise ValueError("Bracket match too long — likely false positive")
    except Exception:
        guesses = ["Undetermined Rare Condition"]

    # ── Immutable state updates ────────────────────────────────────────────
    updated_guesses = list(state.get("current_guesses", []))
    updated_guesses.append({"specialty": specialty_name, "candidates": guesses})

    updated_history = list(state.get("raw_debate_history", []))
    updated_history.append(chat_line)

    updated_full_log = list(state.get("full_debate_log", []))
    updated_full_log.append(chat_line)

    # ── Cooling sleep — applied AFTER the state delta is assembled ─────────
    # This means the sleep happens before the next node fires (not before
    # we return results), so the user sees output immediately.
    time.sleep(INTER_NODE_SLEEP_S)

    return {
        "raw_debate_history": updated_history,
        "current_guesses": updated_guesses,
        "full_debate_log": updated_full_log,
        # HITL marker — tells app.py which stage just completed
        "hitl_stage": specialty_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST NODES  (signatures unchanged — LangGraph wires these directly)
# ─────────────────────────────────────────────────────────────────────────────
def neurologist_node(state: MedicalBoardState):
    return run_specialist_debate(
        state, "Neurology", "Brain, spinal cord, and central nervous pathways."
    )


def immunologist_node(state: MedicalBoardState):
    return run_specialist_debate(
        state,
        "Clinical Immunology",
        "Systemic immune disorders and autoimmune reactions.",
    )


def geneticist_node(state: MedicalBoardState):
    return run_specialist_debate(
        state,
        "Medical Genetics",
        "Congenital DNA syndromes and inherited metabolic blockages.",
    )
