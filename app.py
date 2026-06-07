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

# Browser tab presentation configurations
st.set_page_config(page_title="House M.D. Swarm", layout="wide")
st.title("🩺 The 'House M.D.' Swarm: Rare Disease Explorer")
st.caption("An Autonomous Stateful Multi-Agent Deliberation Panel for Complex Clinical Diagnostics")

def build_workflow_graph():
    # Initialize the architecture state tracker graph
    workflow = StateGraph(MedicalBoardState)
    
    # Register graph computational nodes
    workflow.add_node("parser", run_parser_node)
    workflow.add_node("neurologist", neurologist_node)
    workflow.add_node("immunologist", immunologist_node)
    workflow.add_node("geneticist", geneticist_node)
    workflow.add_node("scribe", run_scribe_node)
    workflow.add_node("research", run_research_node)
    workflow.add_node("cmo", run_cmo_node)
    
    # Draw linear structural execution paths
    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "neurologist")
    workflow.add_edge("neurologist", "immunologist")
    workflow.add_edge("immunologist", "geneticist")
    workflow.add_edge("geneticist", "scribe")
    
    # Establish looping criteria links using your router file configurations
    workflow.add_conditional_edges(
        "scribe",
        evaluate_convergence_edge,
        {
            "continue": "neurologist",
            "research": "research"
        }
    )
    
    workflow.add_edge("research", "cmo")
    workflow.add_edge("cmo", END)
    
    return workflow.compile()

# Build the document file ingestion card
uploaded_file = st.file_uploader("Upload Patient Case Report (PDF Format)", type=["pdf"])

if uploaded_file is not None:
    # Run a fast localized page text reader layout sequence
    pdf_reader = PdfReader(uploaded_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text() + "\n"
        
    st.success("Case file text extracted successfully!")
    
    if st.button("Trigger Autonomous Diagnostics Board"):
        # Setup initial clipboard dictionary variables
        initial_state: MedicalBoardState = {
            "raw_narrative": extracted_text,
            "validated_hpo_codes": [],
            "hpo_labels": {},
            "current_guesses": [],
            "raw_debate_history": [],
            "full_debate_log": [],  # Initialized empty for transcript logging
            "compressed_transcript": "No notes yet. Debate has initialized.",
            "evidence_payload": {},
            "final_report": {},
            "debate_turn_counter": 0
        }
        
        # Compile and invoke the system brain loop execution framework
        app_graph = build_workflow_graph()
        
        with st.spinner("Medical board convening... analyzing data vectors via Groq..."):
            final_output = app_graph.invoke(initial_state)
            
        st.balloons()
        st.markdown("---")
        
        # Split layout presentation cards
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.subheader("💬 Live Medical Boardroom Transcript")
            
            # Loop through permanent records and output with custom avatars
            for line in final_output.get("full_debate_log", []):
                if "[Neurology Doctor]" in line:
                    clean_text = line.replace("[Neurology Doctor]: ", "")
                    with st.chat_message("assistant", avatar="🧠"):
                        st.markdown(f"**Neurologist:** {clean_text}")
                        
                elif "[Clinical Immunology Doctor]" in line:
                    clean_text = line.replace("[Clinical Immunology Doctor]: ", "")
                    with st.chat_message("assistant", avatar="🛡️"):
                        st.markdown(f"**Immunologist:** {clean_text}")
                        
                elif "[Medical Genetics Doctor]" in line:
                    clean_text = line.replace("[Medical Genetics Doctor]: ", "")
                    with st.chat_message("assistant", avatar="🧬"):
                        st.markdown(f"**Geneticist:** {clean_text}")
        
        with right_col:
            st.subheader("📋 Chief Medical Officer Diagnostic Synthesis")
            report = final_output.get("final_report", {})
            if report:
                st.write(report.get("free_text_report", "No report text synthesized."))
            else:
                st.error("No candidate diagnoses survived the verification filtering limits.")
