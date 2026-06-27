"""
app.py  —  House M.D. Swarm: Streamlit front-end (fully autonomous edition)
============================================================================

Single-click run: one button triggers the entire graph to completion.
The UI updates live after each node finishes (via st.empty placeholders),
so you can watch the debate unfold without clicking anything.
Partial output is preserved in session_state if an error occurs mid-run.
"""

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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="House M.D. Swarm", layout="wide")
st.title("🩺 The 'House M.D.' Swarm: Rare Disease Explorer")
st.caption("Autonomous Multi-Agent Deliberation Panel for Complex Clinical Diagnostics")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_workflow_graph():
    workflow = StateGraph(MedicalBoardState)
    workflow.add_node("parser", run_parser_node)
    workflow.add_node("neurologist", neurologist_node)
    workflow.add_node("immunologist", immunologist_node)
    workflow.add_node("geneticist", geneticist_node)
    workflow.add_node("scribe", run_scribe_node)
    workflow.add_node("research", run_research_node)
    workflow.add_node("cmo", run_cmo_node)

    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "neurologist")
    workflow.add_edge("neurologist", "immunologist")
    workflow.add_edge("immunologist", "geneticist")
    workflow.add_edge("geneticist", "scribe")
    workflow.add_conditional_edges(
        "scribe",
        evaluate_convergence_edge,
        {"continue": "neurologist", "research": "research"},
    )
    workflow.add_edge("research", "cmo")
    workflow.add_edge("cmo", END)
    return workflow.compile()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in {
    "graph": None,
    "live_state": None,
    "run_complete": False,
    "run_started": False,
    "error": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# NODE LABEL MAP  (for live progress display only — no gating)
# ─────────────────────────────────────────────────────────────────────────────
NODE_LABELS = {
    "parser":    "🔬 Extracting HPO codes from case narrative…",
    "neurologist":   "🧠 Neurologist deliberating…",
    "immunologist":  "🛡️ Immunologist deliberating…",
    "geneticist":    "🧬 Geneticist deliberating…",
    "scribe":    "📝 Scribe compressing transcript…",
    "research":  "🔭 Research node querying evidence base…",
    "cmo":       "📋 Chief Medical Officer synthesising final report…",
}


# ─────────────────────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def render_debate_log(full_log: list):
    for line in full_log:
        if "[Neurology Doctor]" in line:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown("**Neurologist:** " + line.replace("[Neurology Doctor]: ", ""))
        elif "[Clinical Immunology Doctor]" in line:
            with st.chat_message("assistant", avatar="🛡️"):
                st.markdown("**Immunologist:** " + line.replace("[Clinical Immunology Doctor]: ", ""))
        elif "[Medical Genetics Doctor]" in line:
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown("**Geneticist:** " + line.replace("[Medical Genetics Doctor]: ", ""))


def render_state(state: dict):
    if not state:
        return

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("💬 Boardroom Transcript")
        log = state.get("full_debate_log", [])
        if log:
            render_debate_log(log)
        else:
            st.info("Waiting for specialists…")

        hpo = state.get("validated_hpo_codes", [])
        if hpo:
            labels = state.get("hpo_labels", {})
            st.markdown("---")
            st.subheader("🔬 Validated HPO Codes")
            for code in hpo:
                st.markdown(f"- `{code}` — {labels.get(code, 'unknown')}")

    with right_col:
        guesses = state.get("current_guesses", [])
        if guesses:
            st.subheader("🎯 Disease Candidates (evolving)")
            for entry in guesses:
                st.markdown(f"**{entry['specialty']}:** {', '.join(entry['candidates'])}")

        report = state.get("final_report", {})
        if report:
            st.markdown("---")
            st.subheader("📋 Chief Medical Officer — Final Diagnosis")
            st.success(report.get("free_text_report", "No report synthesised."))

        transcript = state.get("compressed_transcript", "")
        if transcript and transcript != "No notes yet. Debate has initialized.":
            st.markdown("---")
            with st.expander("📝 Compressed Board Notes"):
                st.markdown(transcript)


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Patient Case Report (PDF)", type=["pdf"])

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    extracted_text = "".join(p.extract_text() + "\n" for p in pdf_reader.pages)
    st.success("Case file extracted.")

    if st.button("🚀 Run Full Diagnostic Board", type="primary", disabled=st.session_state.run_started):
        # Reset everything for a fresh run
        st.session_state.live_state = {
            "raw_narrative": extracted_text,
            "validated_hpo_codes": [],
            "hpo_labels": {},
            "current_guesses": [],
            "raw_debate_history": [],
            "full_debate_log": [],
            "compressed_transcript": "No notes yet. Debate has initialized.",
            "evidence_payload": {},
            "final_report": {},
            "debate_turn_counter": 0,
            "hitl_stage": None,
        }
        st.session_state.run_complete = False
        st.session_state.run_started = True
        st.session_state.error = None
        st.session_state.graph = build_workflow_graph()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# AUTONOMOUS RUN LOOP — executes only when run_started=True, run_complete=False
# Drains the entire LangGraph stream in one pass, updating placeholders live.
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.run_started and not st.session_state.run_complete and not st.session_state.error:

    status_box = st.empty()        # live "currently running node" indicator
    output_area = st.empty()       # live partial output, replaced after each node

    try:
        stream = st.session_state.graph.stream(
            st.session_state.live_state,
            stream_mode="updates",
        )

        for event in stream:
            node_name, delta = next(iter(event.items()))

            # Merge delta into accumulated state
            st.session_state.live_state.update(delta)

            # Show which node just finished
            label = NODE_LABELS.get(node_name, f"⚙️ {node_name} running…")
            status_box.info(label)

            # Re-render the full accumulated output after every node
            with output_area.container():
                render_state(st.session_state.live_state)

        # Stream exhausted — run is complete
        st.session_state.run_complete = True
        st.session_state.run_started = False
        status_box.empty()
        st.rerun()

    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.run_started = False
        status_box.empty()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STATIC RENDER — shown after completion or on page re-load
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(
        f"⚠️ Error mid-run: `{st.session_state.error}`\n\n"
        "All output computed before the error is preserved below."
    )

if st.session_state.live_state and (st.session_state.run_complete or st.session_state.error):
    st.markdown("---")
    render_state(st.session_state.live_state)

if st.session_state.run_complete:
    st.balloons()
    st.success("✅ Diagnostic board complete.")

    # Reset button for a new case
    if st.button("🔄 Run Another Case"):
        for k in ["live_state", "run_complete", "run_started", "error", "graph"]:
            st.session_state[k] = None if k in ("live_state", "error", "graph") else False
        st.rerun()
