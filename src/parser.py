"""
src/parser.py  —  Phase 1: Lightweight Symptom Parser Node
===========================================================
Strips raw patient narratives via LLM and maps symptoms to validated
Human Phenotype Ontology (HPO) codes from data/hp-base.json.

All API calls route through src.api_utils.generate_with_retry so that
pacing, retry logic, and model selection are managed in one place.
"""

import json
import re

from google.genai import types

from src.api_utils import ACTIVE_MODEL, generate_with_retry
from src.state import MedicalBoardState


# ─────────────────────────────────────────────────────────────────────────────
# HPO LOOKUP  (loaded once at import time, not on every node invocation)
# ─────────────────────────────────────────────────────────────────────────────
def _load_hpo_index(path: str = "data/hp-base.json") -> dict[str, str]:
    """Return {label_lowercase: HP:XXXXXXX} mapping for containment matching."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # Normalise: accept both {id, label} list and {label: id} dict formats
    if isinstance(raw, list):
        return {entry["label"].lower(): entry["id"] for entry in raw}
    return {k.lower(): v for k, v in raw.items()}


_HPO_INDEX: dict[str, str] = _load_hpo_index()


# ─────────────────────────────────────────────────────────────────────────────
# PARSER NODE
# ─────────────────────────────────────────────────────────────────────────────
def run_parser_node(state: MedicalBoardState) -> dict:
    """
    LangGraph node: Phase 1 — extract symptom keywords from the raw narrative,
    then gate them against the local HPO ontology.

    Returns a partial state dict with:
      validated_hpo_codes  : list of matched HP:XXXXXXX codes
      hpo_labels           : {code: human-readable label}
    """
    prompt = f"""
You are a clinical NLP specialist. Extract every distinct medical symptom or
clinical sign from the patient story below. Return ONLY a JSON array of
lowercase symptom strings and nothing else — no explanation, no markdown fences.

Example output: ["muscle weakness", "seizures", "elevated creatine kinase"]

Patient Story:
{state['raw_narrative']}
"""

    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a medical NLP extraction engine. "
            "Respond only with a raw JSON array. No prose, no markdown."
        ),
        temperature=0.1,   # low temperature for deterministic extraction
    )

    # ── Single API call — uses shared pacing + retry wrapper ─────────────────
    response = generate_with_retry(contents=prompt, config=config)
    raw_text = response.text.strip()

    # ── Parse the JSON array the model returned ───────────────────────────────
    # Strip accidental markdown fences the model might sneak in despite instructions
    raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`")
    try:
        symptom_keywords: list[str] = json.loads(raw_text)
    except json.JSONDecodeError:
        # Graceful degradation: pull any quoted strings out of the response
        symptom_keywords = re.findall(r'"([^"]+)"', raw_text)

    # ── Ontological hard-gating: map to HPO codes via containment filter ──────
    validated_codes: list[str] = []
    hpo_labels: dict[str, str] = {}

    for keyword in symptom_keywords:
        keyword_lower = keyword.lower().strip()
        for label, code in _HPO_INDEX.items():
            if keyword_lower in label or label in keyword_lower:
                if code not in validated_codes:
                    validated_codes.append(code)
                    hpo_labels[code] = label
                break  # first match wins per keyword

    # Fallback: avoid passing an empty payload to the debate phase
    if not validated_codes:
        validated_codes = ["HP:0000001"]   # HPO root "Phenotypic abnormality"
        hpo_labels["HP:0000001"] = "phenotypic abnormality (unspecified)"

    return {
        "validated_hpo_codes": validated_codes,
        "hpo_labels": hpo_labels,
    }
