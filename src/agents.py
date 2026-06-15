import os
import time
import logging
from google import genai
from google.genai import types
from google.genai import errors
from src.state import MedicalBoardState

# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON CLIENT
# Instantiate the genai.Client once at module load, not inside every node call.
# Re-creating it on each invocation causes connection-pool churn and wastes
# initialisation overhead on Streamlit Cloud's constrained single-process env.
# ─────────────────────────────────────────────────────────────────────────────
_client = genai.Client()

# ─────────────────────────────────────────────────────────────────────────────
# INTER-AGENT PACING GOVERNOR
# Tracks the wall-clock timestamp of the last successful API call so the retry
# wrapper can enforce a minimum inter-request gap without sleeping unnecessarily
# on the happy path.
# ─────────────────────────────────────────────────────────────────────────────
_last_call_ts: float = 0.0
MIN_INTER_CALL_GAP_SEC: float = 4.0   # tunable: raise to 6–8 if still hitting 429


def _pace():
    """Sleep only the remaining fraction of the minimum inter-call gap."""
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    wait = MIN_INTER_CALL_GAP_SEC - elapsed
    if wait > 0:
        time.sleep(wait)


def _mark_call():
    global _last_call_ts
    _last_call_ts = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# STATUS-CODE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_status_code(exc: errors.APIError) -> int:
    """
    Safely extract the HTTP status code from any google-genai APIError.
    ClientError, ServerError, etc. all inherit from APIError; the attribute
    name differs slightly across SDK patch versions, so we check several paths.
    """
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # Fall back: parse from the string representation
    msg = str(exc)
    for token in msg.split():
        token = token.strip(".:,")
        if token.isdigit():
            code = int(token)
            if 400 <= code < 600:
                return code
    return 500  # unknown server-side error; treat as retryable


def _is_retryable(status_code: int) -> bool:
    """
    Only retry on transient conditions:
      429 – rate limited (RPM / RPD / TPM quota)
      500 – internal server error
      502 – bad gateway
      503 – service unavailable
      504 – gateway timeout
    Hard 4xx errors (400 bad request, 401 auth, 403 forbidden, 404 not found)
    will NEVER succeed on retry and we should surface them immediately.
    """
    return status_code in {429, 500, 502, 503, 504}


# ─────────────────────────────────────────────────────────────────────────────
# RETRY WRAPPER  (replaces the original generate_with_retry)
# ─────────────────────────────────────────────────────────────────────────────
def generate_with_retry(
    model: str,
    contents: str,
    config: types.GenerateContentConfig | None = None,
    max_retries: int = 6,
    base_delay: float = 10.0,
    max_delay: float = 120.0,
) -> types.GenerateContentResponse:
    """
    Calls the Gemini API with:
      • Minimum inter-call pacing (MIN_INTER_CALL_GAP_SEC)
      • Exponential backoff capped at max_delay seconds
      • Immediate re-raise for non-retryable 4xx errors so we don't waste
        retry budget on errors that can never recover (e.g. 400 bad request)
      • Respects Retry-After header when present (429 responses sometimes
        include it in the google-genai SDK exception attributes)

    NOTE: `client` parameter removed – uses the module-level singleton instead.
    """
    delay = base_delay

    for attempt in range(max_retries):
        _pace()  # enforce minimum gap between calls
        try:
            response = _client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            _mark_call()
            return response

        except (errors.ClientError, errors.ServerError, errors.APIError) as exc:
            status_code = _get_status_code(exc)

            if not _is_retryable(status_code):
                # Hard failure – no point retrying (auth error, bad payload, etc.)
                logging.error(
                    f"[API Fatal Error – HTTP {status_code}] Non-retryable. "
                    f"Raising immediately. Detail: {exc}"
                )
                raise

            if attempt == max_retries - 1:
                logging.error(
                    f"[API Error – HTTP {status_code}] All {max_retries} retries "
                    f"exhausted. Raising. Detail: {exc}"
                )
                raise

            # Check for Retry-After hint from the API (429 sometimes provides it)
            retry_after = getattr(exc, "retry_after", None) or getattr(
                exc, "details", {}
            )
            if isinstance(retry_after, dict):
                retry_after = retry_after.get("retry_after", None)
            sleep_for = float(retry_after) if retry_after else delay

            logging.warning(
                f"[API Error – HTTP {status_code}] Attempt {attempt + 1}/{max_retries}. "
                f"Retrying in {sleep_for:.1f}s…"
            )
            time.sleep(sleep_for)
            delay = min(delay * 2, max_delay)  # cap to avoid multi-minute waits
            _mark_call()  # reset pacing timer after a backoff sleep


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DEBATE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_specialist_debate(
    state: MedicalBoardState,
    specialty_name: str,
    specialty_focus: str,
) -> dict:
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

    # ── API call (no `client` argument; uses module singleton now) ────────────
    response = generate_with_retry(
        model="gemini-2.5-flash",          # ← corrected: gemini-3.5-flash does not exist
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
        ),
    )

    argument_output = response.text
    chat_line = f"[{specialty_name} Doctor]: {argument_output}"

    # ── Parse final guesses (unchanged format contract) ──────────────────────
    try:
        guess_str = argument_output.split("[")[-1].split("]")[0]
        guesses = [g.strip() for g in guess_str.split(",")]
        # Guard: reject the parse if it swallowed the whole output
        if len(guesses) == 1 and len(guesses[0]) > 120:
            raise ValueError("Parse captured too much text – likely false bracket match")
    except Exception:
        guesses = ["Undetermined Rare Condition"]

    # ── Immutable state update pattern (no in-place mutation) ────────────────
    updated_guesses = list(state.get("current_guesses", []))
    updated_guesses.append({"specialty": specialty_name, "candidates": guesses})

    updated_history = list(state.get("raw_debate_history", []))
    updated_history.append(chat_line)

    updated_full_log = list(state.get("full_debate_log", []))
    updated_full_log.append(chat_line)

    return {
        "raw_debate_history": updated_history,
        "current_guesses": updated_guesses,
        "full_debate_log": updated_full_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST NODES  (signatures unchanged – LangGraph wires these directly)
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
