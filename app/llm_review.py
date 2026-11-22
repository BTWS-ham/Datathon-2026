"""LLM review of the facilities shown on a results page.

The review is a second opinion next to the rule-based evidence level. It never changes the
ranking. Each review reads the facility's own text and must quote one line word for word;
a review whose quote is not found in the text is discarded.

Prompt adapted from the 50-pair pilot (06_llm_verification_pilot): LLM 44/50 vs rules 42/50
against human labels. The pilot showed the LLM tends to upgrade machine-only mentions to
"strong", so the label definitions below spell that case out and ask for the lower label
when unsure. That wording has not been re-evaluated on a fresh sample.

APP_LLM=off disables reviews; APP_LLM=stub uses a deterministic fake for tests.
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

MODEL = os.getenv("SERVING_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
LEVELS = ("strong", "moderate", "weak", "none")
RANK = {level: i for i, level in enumerate(reversed(LEVELS))}  # none=0 ... strong=3

SERVICE_DEF = {
    "dialysis": "Dialysis (haemodialysis or peritoneal dialysis / CAPD)",
    "ct_scan": "CT scan",
    "mri": "MRI scan",
    "icu": "ICU (intensive care)",
    "nicu": "NICU (newborn intensive care; general newborn care or jaundice treatment is NOT a NICU)",
    "c_section": "C-section (caesarean delivery)",
    "emergency_trauma": "Emergency and trauma care (emergency dental care does NOT count)",
    "cardiac_surgery": "Cardiac surgery (cardiology, cath lab, angioplasty or gastric bypass do NOT count)",
    "burn_care": "Burn care (sunburn or cosmetic treatment does NOT count)",
}

PROMPT = """You are checking one listing in an Indian healthcare facility directory for a community health worker. Using ONLY the facility text below, decide whether THIS facility currently provides the service on site. Do not use outside knowledge about the facility.

Service: <<SERVICE>>

Labels:
- strong: a line explicitly says the facility itself provides, performs, offers, or has a unit, facility or service for this (e.g. "Provides dialysis treatment", "CT scan available", "NICU available").
- moderate: only a machine, scanner or system is named (e.g. "CT scanner", "MRI imaging system used for diagnosis", "Radiology department equipped with MRI"), with no line saying the service is offered.
- weak: only a related specialty or department is mentioned (e.g. "nephrology", "cardiology" for cardiac surgery, "Radiology department").
- none: nothing relevant, or the only mentions are future or planned ("being set up", "coming soon"), a referral to another facility, outsourced, or unavailable.
If unsure between two labels, choose the lower one.

Facility text (one claim per line, field name in brackets):
<<TEXT>>

Answer with ONLY a JSON object, no other text:
{"level": "strong|moderate|weak|none", "quote": "the single line that best supports your label, copied exactly without the bracketed field name; for none, the line showing it is planned, referred or unavailable; empty string only if nothing relevant exists", "reason": "one or two plain-English sentences for a health worker; mention limits such as inpatient-only, planned, or several branches merged in one record"}"""


def build_prompt(service: str, lines: list[tuple[str, str]]) -> str:
    text = "\n".join(f"[{field}] {line}" for field, line in lines)
    return PROMPT.replace("<<SERVICE>>", SERVICE_DEF.get(service, service.replace("_", " "))).replace("<<TEXT>>", text)


def parse_answer(raw: str, lines: list[tuple[str, str]]) -> dict:
    """Parse the model's JSON and check its quote against the facility text."""
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return {"error": "the AI answer was not valid JSON"}
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return {"error": "the AI answer was not valid JSON"}
    level = str(data.get("level", "")).strip().lower()
    if level not in LEVELS:
        return {"error": f"the AI returned an unknown label ({level or 'empty'})"}
    quote = str(data.get("quote") or "").strip()
    if quote and not any(quote in line for _, line in lines):
        return {"error": "the AI quoted text that is not in this listing, so its review was discarded"}
    return {"level": level, "quote": quote, "reason": str(data.get("reason") or "").strip(), "model": MODEL}


# Words showing that a service exists, not just a machine or department. An AI upgrade over the
# rules is only accepted when its quote contains one of these. Derived from the 50-pair pilot:
# it keeps the 2 correct upgrades and blocks the 4 machine-only over-claims (same sample, so
# it may be optimistic).
AVAILABILITY = re.compile(
    r"\b(available|availability|provides?|provided|providing|offers?|offered|offering|performs?|performed|"
    r"performing|facility|facilities|unit|units|services?|centre|center|on-?site|in-house)\b", re.I)


def reconcile(rule_level: str, review: dict) -> dict:
    """Decide which rating to recommend when the rules and the AI disagree.

    Returns {"agree": bool, "recommended": level, "why": sentence}. AI downgrades are accepted
    (the AI reads context such as "planned" or "inpatients only"); AI upgrades are accepted only
    if its quote states that the service exists. Without a quote the rule rating is kept.
    """
    ai = review["level"]
    if ai == rule_level:
        return {"agree": True, "recommended": ai, "why": ""}
    if not review.get("quote"):
        return {"agree": False, "recommended": rule_level,
                "why": "the AI gave no supporting line from the listing, so the rule rating is kept."}
    if RANK[ai] < RANK[rule_level]:
        return {"agree": False, "recommended": ai,
                "why": "the AI read context in the quoted line that keyword rules cannot, such as a "
                       "planned service, a referral or a restriction."}
    if AVAILABILITY.search(review["quote"]):
        return {"agree": False, "recommended": ai,
                "why": "the quoted line says the service is available or offered, not just that "
                       "equipment or a department exists."}
    return {"agree": False, "recommended": rule_level,
            "why": "the quoted line only names equipment or a department, which does not show the "
                   "service is offered, so the rule rating is kept."}


class Reviewer:
    """Calls a Databricks model serving endpoint; results are cached per facility and service."""

    def __init__(self):
        from databricks.sdk import WorkspaceClient
        self.client = WorkspaceClient()
        self._cache: dict = {}
        self._lock = threading.Lock()

    def _ask(self, prompt: str) -> str:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        resp = self.client.serving_endpoints.query(
            name=MODEL, messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
            temperature=0.0, max_tokens=200)
        return resp.choices[0].message.content

    def review(self, key: tuple, service: str, lines: list[tuple[str, str]]) -> dict:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        try:
            result = parse_answer(self._ask(build_prompt(service, lines)), lines)
        except Exception as e:  # network, permission or quota problems: show, don't crash
            return {"error": f"the AI service could not be reached ({type(e).__name__})"}
        with self._lock:
            self._cache[key] = result
        return result


class StubReviewer:
    """Deterministic stand-in used by tests (APP_LLM=stub)."""

    def review(self, key: tuple, service: str, lines: list[tuple[str, str]]) -> dict:
        def answer(level, line, reason):
            return {"level": level, "quote": line, "model": "stub", "reason": reason}
        for field, line in lines:
            if "being set up" in line.lower():
                return answer("none", line, "The unit is still being set up, so it is not available yet.")
        for field, line in lines:
            if "inpatient" in line.lower():
                return answer("moderate", line, "The unit is described for inpatients only, so outpatients may not be accepted.")
        for field, line in lines:
            if field in ("procedure", "capability"):
                return answer("strong", line, "The listing states the service.")
        for field, line in lines:
            if field == "equipment":
                return answer("strong", line, "The equipment line suggests the service is offered.")
        if any("radiology" in line.lower() for _, line in lines):
            return parse_answer('{"level": "strong", "quote": "Not in the listing", "reason": "x"}', lines)
        return answer("weak", lines[0][1] if lines else "", "Only a related department is mentioned.")


def make_reviewer():
    """Return (reviewer or None, note explaining why AI review is off)."""
    mode = os.getenv("APP_LLM", "on").lower()
    if mode == "off":
        return None, "AI review is turned off (APP_LLM=off)."
    if mode == "stub":
        return StubReviewer(), ""
    try:
        return Reviewer(), ""
    except Exception as e:
        return None, f"AI review is unavailable: no Databricks credentials ({type(e).__name__})."


def review_many(reviewer, jobs: list[tuple[tuple, str, list]], workers: int = 10) -> dict:
    """jobs: (key, service, lines). Runs in parallel; returns {key: result}."""
    if not jobs:
        return {}
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        results = pool.map(lambda job: reviewer.review(*job), jobs)
        return {job[0]: result for job, result in zip(jobs, results)}
