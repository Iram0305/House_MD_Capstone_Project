from typing import TypedDict, List, Dict, Any

class MedicalBoardState(TypedDict):
    raw_narrative: str
    validated_hpo_codes: List[str]
    hpo_labels: Dict[str, str]
    current_guesses: List[Dict[str, Any]]
    raw_debate_history: List[str]
    full_debate_log: List[str]         # 👈 ADD THIS LINE to save the entire un-erased chat
    compressed_transcript: str
    evidence_payload: Dict[str, Any]
    final_report: Dict[str, Any]
    debate_turn_counter: int
