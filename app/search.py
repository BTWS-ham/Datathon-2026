"""Bridge between the app and the team ranking in ../ranking.py (04_ranking).

The app never re-implements scoring: every score, evidence level, warning and exclusion
comes from ranking.Ranker, so what users see matches what the team evaluates.
This module only prepares input rows, folds weak evidence into its own group and
attaches display-only fields (facility type, phone, website).
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ranking  # noqa: E402  (team ranking module at the repository root)

SERVICE_LABELS = {
    "dialysis": "Dialysis",
    "hemodialysis": "Dialysis (haemodialysis)",
    "ct_scan": "CT scan",
    "mri": "MRI scan",
    "c_section": "C-section (caesarean delivery)",
    "nicu": "NICU (newborn intensive care)",
    "icu": "ICU (intensive care)",
    "emergency_trauma": "Emergency & trauma care",
    "cardiac_surgery": "Cardiac surgery",
    "burn_care": "Burn care",
}
GENERIC_WARNING = "Facility identity and current service availability are not independently verified."
FLAG_TEXT = {
    "flag_coords_far_from_city": "The map location is far from other facilities in the same city, so the distance may be wrong.",
    "flag_pin_state_mismatch": "PIN code and state disagree; this record may merge several branches.",
}
EXCLUSION_TEXT = {
    "missing_facility_match": "Evidence row has no matching facility record",
    "corrupted_facility": "Record is corrupted (no usable name)",
    "unusable_coordinates": "Coordinates missing or outside India",
    "unusable_identity": "Facility identity is unusable",
    "invalid_evidence": "Evidence row is malformed",
    "missing_structured_quotes": "Evidence has no quotable source sentence",
    "quote_not_in_source": "Quoted evidence was not found in the facility's own text",
    "unknown_corruption_status": "Record quality could not be checked",
}
DETAIL_COLUMNS = ["facility_type", "officialPhone", "officialWebsite", "phone_shared_by_n", "pin_code",
                  "procedure_list", "capability_list", "equipment_list", "description", "specialties_list"]
TEXT_FIELDS = [("procedure_list", "procedure"), ("capability_list", "capability"),
               ("equipment_list", "equipment"), ("description", "description"),
               ("specialties_list", "specialty tag")]


def service_label(service: str) -> str:
    return SERVICE_LABELS.get(service, service.replace("_", " ").capitalize())


def _native(value):
    """Convert pandas / numpy scalars to plain Python values (ranking checks `type(x) is bool`)."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (list, dict, str)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _native(v) for k, v in row.items()} for row in df.to_dict("records")]


def build_ranker(facilities: pd.DataFrame, evidence: pd.DataFrame) -> ranking.Ranker:
    """Same input handling as ranking.load_from_spark, but from pandas frames."""
    frows = _records(facilities)
    counts = Counter(str(r.get("unique_id") or "").strip() for r in frows)
    ambiguous = {uid for uid, n in counts.items() if uid and n > 1}
    usable = [r for r in frows if str(r.get("unique_id") or "").strip() not in ambiguous]
    erows = _records(evidence.rename(columns={"quotes_json": "quotes"}))
    ranker = ranking.Ranker(usable, erows)
    ranker.input_diagnostics = {
        "ambiguous_facility_ids_excluded": len(ambiguous),
        "facility_rows_excluded_for_ambiguous_ids": sum(counts[uid] for uid in ambiguous),
    }
    return ranker


def _url_list(value) -> list[str]:
    if isinstance(value, list):
        return [u for u in value if isinstance(u, str)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [u for u in parsed if isinstance(u, str)] if isinstance(parsed, list) else []
        except ValueError:
            return [value] if value.startswith("http") else []
    return []


class Search:
    def __init__(self, facilities: pd.DataFrame, evidence: pd.DataFrame):
        self.ranker = build_ranker(facilities, evidence)
        cols = ["unique_id"] + [c for c in DETAIL_COLUMNS if c in facilities.columns]
        details = facilities[cols].drop_duplicates("unique_id")
        self.details = {str(r["unique_id"]).strip(): r for r in _records(details)}

    @property
    def services(self) -> list[str]:
        return sorted(self.ranker.evidence, key=service_label)

    def facility_lines(self, unique_id: str) -> list[tuple[str, str]]:
        """The facility's own text as (field, line) pairs, de-duplicated, for the AI review."""
        d = self.details.get(str(unique_id).strip(), {})
        out, seen = [], set()
        for col, field in TEXT_FIELDS:
            value = d.get(col)
            if isinstance(value, str) and value.strip().startswith("["):
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            items = value if isinstance(value, list) else ([value] if value else [])
            for item in items:
                line = str(item).strip()
                if line and line.lower() not in seen:
                    seen.add(line.lower())
                    out.append((field, line))
        return out

    def _card(self, c: dict) -> dict:
        card = dict(c)
        d = self.details.get(c["unique_id"], {})
        card["facility_type"] = d.get("facility_type")
        card["phone"] = d.get("officialPhone")
        card["website"] = d.get("officialWebsite")
        card["phone_shared_by_n"] = int(d.get("phone_shared_by_n") or 0)
        card["source_urls"] = _url_list(c.get("source_urls"))
        raw = [w for w in c.get("data_warnings", [])
               if w != GENERIC_WARNING and not w.startswith("The cleaner labels location confidence")]
        notes = [p.strip() for w in raw if w not in FLAG_TEXT for p in str(w).split(";") if p.strip()]
        # Ranking also returns bare flag names; only add their text if quality_notes did not cover them
        covered = {"flag_pin_state_mismatch": "pin code", "flag_coords_far_from_city": "coordinates"}
        for w in raw:
            if w in FLAG_TEXT and not any(covered[w] in n.lower() for n in notes):
                notes.append(FLAG_TEXT[w])
        card["warnings"] = list(dict.fromkeys(notes))
        return card

    def run(self, service: str, lat: float, lon: float, k: int = 5) -> dict:
        res = self.ranker.recommend(service, lat, lon, k=20)
        nearby = res["recommendations"]                      # nearby, in score order
        main, seen = [], set()
        for c in nearby:
            key = (ranking_key(c["name"]), ranking_key(c["city"]))
            if c["evidence_level"] != "weak" and key not in seen:
                seen.add(key)
                main.append(self._card(c))
        weak = [self._card(c) for c in res["nearby_weak_options"]]
        nearest = res["nearest_strong_option"]
        shown = {c["unique_id"] for c in main[:k]}
        nearest = self._card(nearest) if nearest and res["nearby_strong_count"] == 0 and nearest["unique_id"] not in shown else None
        excluded = {EXCLUSION_TEXT.get(key, key): n for key, n in res["excluded_rows"].items()}
        return {
            "service": res["service"], "status": res["status"], "message": res["message"],
            "policy": res["policy"], "main": main[:k], "weak": weak[:k], "n_weak": len(weak),
            # A few extra candidates so the list can still be filled if the AI review removes some
            "pool": main[:k + 3], "k": k,
            "nearest_strong": nearest, "nearby_strong_count": res["nearby_strong_count"],
            "eligible": res["eligible_candidate_count"], "excluded": excluded,
            "n_evidence_rows": len(self.ranker.evidence.get(res["service"], [])),
            "diagnostics": res.get("input_diagnostics", {}),
        }


def ranking_key(value) -> str:
    return " ".join(str(value or "").lower().split())
