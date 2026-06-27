"""
src/api_utils.py  —  Shared Gemini API utility layer
=====================================================
Single source of truth for:
  • The singleton genai.Client (one object for the entire process lifetime)
  • The inter-call pacing governor  (prevents burst 429s)
  • The generate_with_retry wrapper (handles transient errors & Retry-After)
  • Automatic model fallback chain (primary → fallback → last-resort)

Import from here in BOTH parser.py AND agents.py so the logic never drifts.

Free-tier RPM reality check
----------------------------
gemini-2.0-flash  → 15 RPM  = 1 req / 4s  (1,500 req/day)
gemini-1.5-flash  → 15 RPM  = 1 req / 4s  (1,500 req/day)
gemini-1.5-flash-8b → 15 RPM (separate quota bucket — used as last-resort)

Your swarm fires 7–21 calls per run. The pacing governor enforces a hard
floor of 6s between ALL calls (= max 10 RPM), keeping you safely below the
15 RPM ceiling even during the cyclic debate loop.

When a model's own quota bucket is exhausted, the fallback chain rotates to
the next model automatically instead of burning all retries on a dead quota.
"""

import json
import logging
import time

from google import genai
from google.genai import errors, types

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# Three separate free-tier quota buckets. When the primary is rate-limited,
# generate_with_retry() falls through to the next one automatically.
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_MODEL = "gemini-2.0-flash"          # primary   — 1,500 req/day free

MODEL_FALLBACK_CHAIN = [
    "gemini-2.0-flash",                    # primary
    "gemini-1.5-flash",                    # fallback  — separate quota bucket
    "gemini-1.5-flash-8b",                 # last-resort — smallest, most lenient
]

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

# 6s gap = max 10 RPM sustained, safely under the 15 RPM free-tier ceiling.
# The 2.0-flash burst window is a rolling 60s bucket, so staying at 10 RPM
# gives a 33% headroom buffer against any burst spike during the debate loop.
MIN_INTER_CALL_GAP_SEC: float = 6.0


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
    max_retries: int = 3,
    base_delay: float = 6.0,
    max_delay: float = 60.0,
) -> types.GenerateContentResponse:
    """
    Call the Gemini API with pacing, smart retries, and automatic model fallback.

    Retry strategy (per model in the fallback chain):
      • max_retries=3 attempts per model (down from 6 — stops wasting time on
        a quota bucket that's clearly exhausted)
      • base_delay=6.0s aligns with the pacing gap so the first retry fires
        at the same cadence as normal calls — no extra waiting
      • On persistent 429, rotates to the next model in MODEL_FALLBACK_CHAIN
        instead of hammering the same exhausted quota bucket

    Parameters
    ----------
    contents  : The user prompt string.
    config    : Optional GenerateContentConfig (system instruction, temperature…).
    model     : Pin to a specific model, bypassing the fallback chain.
    max_retries, base_delay, max_delay : Retry tuning knobs.

    Raises
    ------
    errors.APIError  after all models in the chain are exhausted.
    """
    # If caller pins a model, use only that model (no chain)
    chain = [model] if model else MODEL_FALLBACK_CHAIN

    last_exc = None

    for current_model in chain:
        delay = base_delay
        for attempt in range(max_retries):
            _pace()
            try:
                response = _client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config,
                )
                _mark_call()
                if current_model != ACTIVE_MODEL:
                    logging.info(
                        f"[API Fallback] Successfully used '{current_model}' "
                        f"(primary '{ACTIVE_MODEL}' was rate-limited)."
                    )
                return response

            except (errors.ClientError, errors.ServerError, errors.APIError) as exc:
                last_exc = exc
                status_code = _get_status_code(exc)

                # Non-retryable hard error — don't try other models either
                if not _is_retryable(status_code):
                    logging.error(
                        f"[API Fatal {status_code}] Non-retryable on "
                        f"'{current_model}'. Detail: {exc}"
                    )
                    raise

                # All retries for this model exhausted → try next model
                if attempt == max_retries - 1:
                    logging.warning(
                        f"[API] {max_retries} retries exhausted on "
                        f"'{current_model}'. Rotating to next model in chain."
                    )
                    break

                retry_after = _get_retry_after(exc)
                sleep_for = retry_after if retry_after else delay

                logging.warning(
                    f"[API Error {status_code}] Attempt {attempt + 1}/{max_retries} "
                    f"on model '{current_model}'. "
                    f"{'Retry-After hint' if retry_after else 'Backoff'}. "
                    f"Sleeping {sleep_for:.1f}s…"
                )
                time.sleep(sleep_for)
                delay = min(delay * 2, max_delay)
                _mark_call()

    # Every model in the chain failed
    logging.error(
        "[API] All models in fallback chain exhausted. "
        f"Chain tried: {chain}. Last error: {last_exc}"
    )
    raise last_exc
