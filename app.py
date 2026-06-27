"""
app.py  —  House M.D. Swarm: Streamlit front-end (HITL streaming edition)
==========================================================================

Architecture changes vs. original:
  1. app_graph.stream() replaces app_graph.invoke() — every node's state
     delta is written into st.session_state as it arrives, so a mid-run
     crash never hides already-computed output.
  2. HITL stage gates: after each significant node the stream loop
     writes a `pending_approval` key and immediately re-runs Streamlit.
     The UI renders the completed output and a "Proceed" button.  Nothing
     token-consuming fires again until the user clicks.
  3. The graph is compiled once and stored in session_state so it survives
     the Streamlit re-run cycle triggered by button clicks.
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
st.caption(
    "An Autonomous Stateful Multi-Agent Deliberation Panel for Complex Clinical Diagnostics"
)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH FACTORY  (compiled once, stored in session_state)
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
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "graph": None,
        "live_state": None,          # MedicalBoardState built up incrementally
        "stream_iter": None,         # Active LangGraph stream iterator
        "pending_approval": None,    # Stage name waiting for user click
        "run_complete": False,
        "error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ─────────────────────────────────────────────────────────────────────────────
# HITL GATE DEFINITIONS
# Maps node names → display labels shown above the "Proceed" button.
# ─────────────────────────────────────────────────────────────────────────────
HITL_GATES = {
    "parser":            "🔬 Parser complete — HPO codes extracted",
    "Neurology":         "🧠 Neurologist has submitted their analysis",
    "Clinical Immunology": "🛡️ Immunologist has submitted their analysis",
    "Medical Genetics":  "🧬 Geneticist has submitted their analysis",
    "scribe":            "📝 Scribe has compressed the board transcript",
    "research":          "🔭 Research node complete — evidence retrieved",
}

# Nodes that should pause for human approval before proceeding
PAUSE_AFTER = set(HITL_GATES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_debate_log(full_log: list):
    """Render the debate transcript with per-specialist chat bubbles."""
    for line in full_log:
        if "[Neurology Doctor]" in line:
            text = line.replace("[Neurology Doctor]: ", "")
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(f"**Neurologist:** {text}")
        elif "[Clinical Immunology Doctor]" in line:
            text = line.replace("[Clinical Immunology Doctor]: ", "")
            with st.chat_message("assistant", avatar="🛡️"):
                st.markdown(f"**Immunologist:** {text}")
        elif "[Medical Genetics Doctor]" in line:
            text = line.replace("[Medical Genetics Doctor]: ", "")
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(f"**Geneticist:** {text}")


def render_partial_state(state: dict):
    """
    Render whatever has been computed so far.
    Called both during the stream loop and on every re-run.
    """
    if not state:
        return

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("💬 Live Boardroom Transcript")
        log = state.get("full_debate_log", [])
        if log:
            render_debate_log(log)
        else:
            st.info("Waiting for specialists to speak...")

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
            st.subheader("🎯 Current Disease Candidates")
            for entry in guesses:
                st.markdown(f"**{entry['specialty']}:** {', '.join(entry['candidates'])}")

        report = state.get("final_report", {})
        if report:
            st.markdown("---")
            st.subheader("📋 Chief Medical Officer Synthesis")
            st.write(report.get("free_text_report", "No report text synthesized."))

        transcript = state.get("compressed_transcript", "")
        if transcript and transcript != "No notes yet. Debate has initialized.":
            st.markdown("---")
            with st.expander("📝 Compressed Board Notes (Scribe)"):
                st.markdown(transcript)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def advance_stream():
    """
    Pull the next event from the LangGraph stream iterator.
    Merges the state delta into session_state.live_state.
    If the completed node is in PAUSE_AFTER, sets pending_approval and stops.
    Catches all exceptions so partial output is never lost.
    """
    iterator = st.session_state.stream_iter
    if iterator is None:
        return

    try:
        event = next(iterator)                      # {node_name: state_delta}
    except StopIteration:
        st.session_state.run_complete = True
        st.session_state.stream_iter = None
        return
    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.stream_iter = None
        return

    # Merge the delta into our accumulated live_state
    node_name, delta = next(iter(event.items()))
    if st.session_state.live_state is None:
        st.session_state.live_state = {}
    st.session_state.live_state.update(delta)

    # Determine the stage label for HITL gating:
    # agents write hitl_stage = specialty_name; other nodes use node_name.
    stage_key = delta.get("hitl_stage") or node_name

    if stage_key in PAUSE_AFTER:
        st.session_state.pending_approval = stage_key
    else:
        # Non-gated node: keep streaming automatically
        advance_stream()


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD & RUN TRIGGER
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Patient Case Report (PDF Format)", type=["pdf"]
)

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    extracted_text = "".join(
        page.extract_text() + "\n" for page in pdf_reader.pages
    )
    st.success("Case file text extracted successfully!")

    if (
        st.button("Trigger Autonomous Diagnostics Board")
        and st.session_state.stream_iter is None
        and not st.session_state.run_complete
    ):
        initial_state: MedicalBoardState = {
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

        if st.session_state.graph is None:
            st.session_state.graph = build_workflow_graph()

        # Open the stream — nothing executes yet; first advance_stream() call
        # will pull the first event.
        st.session_state.stream_iter = st.session_state.graph.stream(
            initial_state, stream_mode="updates"
        )
        st.session_state.live_state = dict(initial_state)
        st.session_state.run_complete = False
        st.session_state.error = None
        st.session_state.pending_approval = None

        # Kick off streaming until the first HITL gate
        advance_stream()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI — renders on every Streamlit re-run
# ─────────────────────────────────────────────────────────────────────────────
state = st.session_state.live_state

# ── Error banner (partial output already shown below) ─────────────────────
if st.session_state.error:
    st.error(
        f"⚠️ An error occurred: `{st.session_state.error}`\n\n"
        "All outputs computed before the error are preserved below."
    )

# ── Partial / final output ─────────────────────────────────────────────────
if state:
    st.markdown("---")
    render_partial_state(state)

# ── HITL approval gate ─────────────────────────────────────────────────────
pending = st.session_state.pending_approval

if pending and not st.session_state.run_complete and not st.session_state.error:
    st.markdown("---")
    gate_label = HITL_GATES.get(pending, f"Stage '{pending}' complete")
    st.info(f"**{gate_label}**\n\nReview the output above, then proceed when ready.")

    if st.button("▶ Proceed to Next Phase", type="primary"):
        st.session_state.pending_approval = None
        advance_stream()           # Pull next event(s) until the next gate
        st.rerun()

# ── Completion banner ──────────────────────────────────────────────────────
if st.session_state.run_complete:
    st.balloons()
    st.success("✅ Diagnostic board session complete. See the CMO report above.")

# ── Active streaming indicator (no gate pending, stream still open) ────────
elif st.session_state.stream_iter is not None and pending is None:
    with st.spinner("Medical board is deliberating..."):
        advance_stream()
        st.rerun()
