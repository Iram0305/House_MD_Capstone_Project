from src.state import MedicalBoardState


def evaluate_convergence_edge(state: MedicalBoardState) -> str:
    current_turn = state.get("debate_turn_counter", 1)
    print(f"Evaluating routing criteria. Current turn tally: {current_turn}")

    # Safety iteration guard
    if current_turn >= 4:  # Lowered to 4 for quicker prototyping runs
        print("Maximum turn loop threshold met. Forwarding directly to Library RAG node.")
        return "research"

    # Beginner fallback: forcing a standard 3-turn multi-specialist critique baseline
    if current_turn < 3:
        return "continue"

    return "research"