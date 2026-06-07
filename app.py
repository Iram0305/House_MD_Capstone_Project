import streamlit as st
from pypdf import PdfReader
from langgraph.graph import StateGraph, END
from src.state import MedicalBoardState
from src.parser import run_parser_node
from src.agents import neurologist_node, immunologist_node, geneticist_node
from src.scribe import run_scribe_node
from src.research import run_research_node
from src.cmo import run_cmo_node
from src.router import evaluate_convergence_edge

# Page configurations
st.set_page_config(page_title="House M.D. Swarm", layout="wide")
st.title("🩺 The 'House M.D.' Swarm: Rare Disease Explorer")
st.caption("An Autonomous Stateful Multi-Agent Deliberation Panel for Complex Clinical Diagnostics")


# 1. Build the LangGraph State Machine Workflow Orchestrator
def build_workflow_graph():
    workflow = StateGraph(MedicalBoardState)

    # Register our processing station nodes
    workflow.add_node("parser", run_parser_node)
    workflow.add_node("neurologist", neurologist_node)
    workflow.add_node("immunologist", immunologist_node)
    workflow.add_node("geneticist", geneticist_node)
    workflow.add_node("scribe", run_scribe_node)
    workflow.add_node("research", run_research_node)
    workflow.add_node("cmo", run_cmo_node)

    # Map out the connections
    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "neurologist")
    workflow.add_edge("neurologist", "immunologist")
    workflow.add_edge("immunologist", "geneticist")
    workflow.add_edge("geneticist", "scribe")

    # Insert our custom looping rule edge after the Scribe updates notes
    workflow.add_conditional_edges(
        "scribe",
        evaluate_convergence_edge,
        {
            "continue": "neurologist",  # Loop back to start another debate turn
            "research": "research"  # Break out and run the librarian tool
        }
    )

    workflow.add_edge("research", "cmo")
    workflow.add_edge("cmo", END)

    return workflow.compile()


# UI Layout Components
uploaded_file = st.file_uploader("Upload Patient Case Report (PDF Format)", type=["pdf"])

if uploaded_file is not None:
    # Read the text out of the uploaded file container
    pdf_reader = PdfReader(uploaded_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text() + "\n"

    st.success("Case file text extracted successfully!")

    if st.button("Trigger Autonomous Diagnostics Board"):
        # Setup initial empty state dictionary clipboard
        initial_state: MedicalBoardState = {
            "raw_narrative": extracted_text,
            "validated_hpo_codes": [],
            "hpo_labels": {},
            "current_guesses": [],
            "raw_debate_history": [],
            "compressed_transcript": "No notes yet. Debate has initialized.",
            "evidence_payload": {},
            "final_report": {},
            "debate_turn_counter": 0
        }

        # Compile and execute the full graph workflow engine end to end
        app_graph = build_workflow_graph()

        with st.spinner("Medical board convening... analyzing data vectors..."):
            final_output = app_graph.invoke(initial_state)

        st.balloons()

        # Display Section Split UI panels
        st.markdown("---")
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("📝 Internal Board Scribe Discussion Records")
            st.info(final_output.get("compressed_transcript", "No logs recorded."))

        with right_col:
            st.subheader("📋 Chief Medical Officer Diagnostic Synthesis")
            report = final_output.get("final_report", {})

            if report:
                st.markdown("### 🏆 Prioritized Differential Diagnosis Candidates")
                for rank, condition in enumerate(report.get("prioritized_differential", []), 1):
                    st.markdown(f"**{rank}. {condition}**")

                st.markdown("### 🔍 Evidential Verification Backing")
                st.write(report.get("evidential_justification"))

                st.markdown("### 🧪 Recommended Secondary Confirmatory Profiling")
                for test in report.get("recommended_confirmatory_tests", []):
                    st.markdown(f"- {test}")
            else:
                st.error("No diagnoses passed the strict library evaluation verification check limits.")