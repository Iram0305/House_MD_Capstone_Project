"""
src/api_utils.py  —  Shared Gemini API utility layer
=====================================================
Single source of truth for:
  • The singleton genai.Client (one object for the entire process lifetime)
  • The inter-call pacing governor  (prevents burst 429s)
  • The generate_with_retry wrapper (handles transient errors & Retry-After)

Import from here in BOTH parser.py AND agents.py so the logic never drifts.

Model selection note
--------------------
gemini-3.5-flash  →  FREE TIER LIMIT: only 20 requests/day  ← causes your crash
gemini-2.0-flash  →  FREE TIER LIMIT: 1,500 requests/day   ← recommended
gemini-1.5-flash  →  FREE TIER LIMIT: 1,500 requests/day   ← fallback alternative

Change ACTIVE_MODEL below to swap across the whole project instantly.
"""

import json
import logging
import time

from google import genai
from google.genai import errors, types

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION  ← change this one constant to affect the entire project
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_MODEL = "gemini-2.0-flash"   # 1,500 req/day free vs 20 for gemini-3.5-flash

# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON CLIENT
# One client object for the entire Streamlit process lifetime.
# Never call genai.Client() inside a node function — it creates a new HTTP
# connection pool on every invocation, burning memory on Streamlit Cloud's
# 1 GB ceiling and adding ~200 ms of TLS handshake overhead per call.
# ─────────────────────────────────────────────────────────────────────────────
_client = genai.Client()


# ─────────────────────────────────────────────────────────────────────────────
# PACING GOVERNOR
# Enforces a minimum wall-clock gap between *all* API calls across all nodes.
# This prevents burst traffic from exhausting the per-minute RPM quota even
# when the daily RPD quota is fine.
# ─────────────────────────────────────────────────────────────────────────────
_last_call_ts: float = 0.0
MIN_INTER_CALL_GAP_SEC: float = 5.0  # raise to 8–12 if you still hit RPM 429s


def _pace() -> None:
    """Sleep only the remaining fraction of the minimum inter-call gap."""
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    gap = MIN_INTER_CALL_GAP_SEC - elapsed
    if gap > 0:
        time.sleep(gap)


def _mark_call() -> None:
    global _last_call_ts
    _last_call_ts = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# STATUS-CODE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_status_code(exc: errors.APIError) -> int:
    """Safely extract HTTP status code from any google-genai APIError subclass."""
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # Last resort: scan the string representation for a 3-digit HTTP code
    for token in str(exc).split():
        token = token.strip(".:,()[]{}")
        if token.isdigit() and 400 <= int(token) < 600:
            return int(token)
    return 500


def _get_retry_after(exc: errors.APIError) -> float | None:
    """
    Extract the Retry-After hint that Google embeds in 429 responses.
    The SDK surfaces it inside the 'details' list as a RetryInfo entry.

    Example detail block:
        {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '9s'}
    """
    try:
        # exc.args[0] is the raw response dict from the API
        raw = exc.args[0] if exc.args else {}
        if isinstance(raw, str):
            raw = json.loads(raw)
        details = raw.get("error", {}).get("details", [])
        for entry in details:
            if "RetryInfo" in entry.get("@type", ""):
                delay_str = entry.get("retryDelay", "")
                # retryDelay is formatted as "9s" or "9.857976303s"
                seconds = float(delay_str.rstrip("s"))
                return seconds + 2.0   # add 2s safety buffer
    except Exception:
        pass
    return None


def _is_retryable(status_code: int) -> bool:
    """
    Only retry on genuinely transient conditions.
    Hard 4xx errors (400, 401, 403, 404) will never recover on retry;
    raising them immediately saves the entire retry budget.

    Note: 429 is a ClientError (4xx) but IS retryable — it's quota, not a bug.
    """
    return status_code in {429, 500, 502, 503, 504}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RETRY WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def generate_with_retry(
    contents: str,
    config: types.GenerateContentConfig | None = None,
    model: str | None = None,
    max_retries: int = 6,
    base_delay: float = 12.0,
    max_delay: float = 120.0,
) -> types.GenerateContentResponse:
    """
    Call the Gemini API with pacing, smart retries, and Retry-After awareness.

    Parameters
    ----------
    contents  : The user prompt string.
    config    : Optional GenerateContentConfig (system instruction, temperature…).
    model     : Override the global ACTIVE_MODEL for this call (rarely needed).
    max_retries, base_delay, max_delay : Retry tuning knobs.

    Raises
    ------
    errors.APIError  on non-retryable errors or after all retries exhausted.
    """
    target_model = model or ACTIVE_MODEL
    delay = base_delay

    for attempt in range(max_retries):
        _pace()   # enforce minimum inter-call gap before every attempt
        try:
            response = _client.models.generate_content(
                model=target_model,
                contents=contents,
                config=config,
            )
            _mark_call()
            return response

        except (errors.ClientError, errors.ServerError, errors.APIError) as exc:
            status_code = _get_status_code(exc)

            # ── Non-retryable: surface immediately, don't waste retry budget ──
            if not _is_retryable(status_code):
                logging.error(
                    f"[API Fatal {status_code}] Non-retryable error on model "
                    f"'{target_model}'. Detail: {exc}"
                )
                raise

            # ── All retries exhausted ─────────────────────────────────────────
            if attempt == max_retries - 1:
                logging.error(
                    f"[API Error {status_code}] All {max_retries} retries "
                    f"exhausted on model '{target_model}'. Raising."
                )
                raise

            # ── Respect the API's own Retry-After hint when present ───────────
            retry_after = _get_retry_after(exc)
            sleep_for = retry_after if retry_after else delay

            logging.warning(
                f"[API Error {status_code}] Attempt {attempt + 1}/{max_retries} "
                f"on model '{target_model}'. "
                f"{'Using API Retry-After hint' if retry_after else 'Using backoff'}. "
                f"Sleeping {sleep_for:.1f}s…"
            )
            time.sleep(sleep_for)
            delay = min(delay * 2, max_delay)
            _mark_call()  # reset pacing timer after sleeping
