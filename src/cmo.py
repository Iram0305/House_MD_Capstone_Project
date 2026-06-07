from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.state import MedicalBoardState


class FinalReportSchema(BaseModel):
    prioritized_differential: list[str] = Field(description="Ranked list of verified rare disease diagnoses.")
    evidential_justification: str = Field(description="Detailed explanation linking symptoms to the citations.")
    recommended_confirmatory_tests: list[str] = Field(description="Exact biological assays, gene panels or scans.")


def run_cmo_node(state: MedicalBoardState) -> dict:
    print("\n--- PHASE 4: CHIEF MEDICAL OFFICER SYNTHESIS ---")

    evidence = state.get("evidence_payload", {})
    verified_context = {}

    # HARD FILTERING RULE: Drop anything that lacks local DB or PubMed evidence support
    for disease, proof in evidence.items():
        if proof["orphadata_match"] or len(proof["pubmed_citations"]) > 0:
            verified_context[disease] = proof
        else:
            print(f"[CMO FILTER RULE TRIGGERED]: Dropped '{disease}' due to zero verified evidence.")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(FinalReportSchema)

    prompt = f"""
    You are the Chief Medical Officer. Read the verified medical evidence and discussion logs.
    Compile the final definitive rare disease diagnostic support report.

    Verified Library Evidence Data:
    {verified_context}

    Case Scribe Discussion Tracking:
    {state['compressed_transcript']}

    Synthesize this data into a structured report mapping diagnoses directly to their citation references.
    """

    report_output = structured_llm.invoke(prompt)

    # Store the output inside our clipboard
    return {"final_report": report_output.model_dump()}