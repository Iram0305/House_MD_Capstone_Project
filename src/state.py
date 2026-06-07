from typing import TypedDict, List, Dict, Any

class MedicalBoardState(TypedDict):
    raw_narrative: str                  # The messy text from the PDF report
    validated_hpo_codes: List[str]      # Cleaned symptom IDs from Phase 1
    hpo_labels: Dict[str, str]          # Human names for those symptom IDs
    current_guesses: List[Dict[str, Any]] # Top disease candidates discussed by doctors
    raw_debate_history: List[str]       # The exact words spoken in the current round
    compressed_transcript: str          # The Scribe's clean summary notes
    evidence_payload: Dict[str, Any]    # Research proofs from Phase 3
    final_report: Dict[str, Any]        # The final document from Phase 4
    debate_turn_counter: int            # Safety counter to stop infinite loops