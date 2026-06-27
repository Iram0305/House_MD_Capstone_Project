from typing import TypedDict, List, Dict, Any, Optional


class MedicalBoardState(TypedDict):
    raw_narrative: str                      # Original unstructured clinical text
    validated_hpo_codes: List[str]          # HPO codes parsed from narrative
    hpo_labels: Dict[str, str]              # Code → human-readable label map
    current_guesses: List[Dict[str, Any]]   # Per-specialist disease candidates
    raw_debate_history: List[str]           # Full uncompressed specialist lines
    full_debate_log: List[str]              # Cumulative log for UI transcript
    compressed_transcript: str             # Scribe-compressed board memory
    evidence_payload: Dict[str, Any]        # Research / RAG retrieval results
    final_report: Dict[str, Any]           # CMO synthesis output
    debate_turn_counter: int               # Cycle counter for convergence router

    # ── HITL stage tracking ────────────────────────────────────────────────
    # Stores the name of the node that most recently completed so the UI
    # knows which "Proceed" gate to render next.  None = not yet started.
    hitl_stage: Optional[str]
