"""Re-rank the shown results with road distances and after the AI review.

The team formula from ranking.py is kept: score = evidence score x location weight x distance
factor. apply_road swaps the straight-line distance for the OSRM road distance inside the distance
factor exp(-distance / decay). When the AI review leads to a different recommended rating (see llm_review.reconcile),
only the evidence score is replaced by the score the evidence rules give that rating; the
location weight and distance factor are unchanged. "none" removes the facility from the list.
Only the reviewed candidates (the shown results plus a few extras) are re-ranked.
"""

from __future__ import annotations

import math

from llm_review import reconcile
from routing import ROAD_FACTOR

# Rule weights used by 03_service_evidence for each evidence level
LEVEL_SCORE = {"strong": 0.85, "moderate": 0.55, "weak": 0.20}
ORDER = lambda c: (-c["score"], c["distance_km"], c["unique_id"])  # noqa: E731  same tie-break as ranking.py


def apply_road(res: dict, roads: dict, error: str | None) -> dict:
    """Use road distance instead of straight-line distance in the distance factor and re-sort.

    Same formula as ranking.py: exp(-distance / decay). Facilities without an OSRM road distance
    (service unavailable or no route) get an estimate, ROAD_FACTOR x straight line, flagged as
    estimated, so every facility is ranked on the same kind of distance.
    """
    decay = res["policy"]["decay_km"]
    straight_rank = {c["unique_id"]: i for i, c in enumerate(res["main"], 1)}

    def update(card):
        road = roads.get(card["unique_id"])
        estimated = road is None
        if estimated:
            road = card["distance_km"] * ROAD_FACTOR
        parts = dict(card["score_breakdown"], distance_factor=math.exp(-road / decay),
                     distance_basis="estimated road" if estimated else "road")
        return dict(card, straight_km=card["distance_km"], road_km=road, road_estimated=estimated,
                    distance_km=road, score_breakdown=parts,
                    score=parts["evidence_score"] * parts["location_weight"] * parts["distance_factor"])

    pool = sorted([update(c) for c in res["pool"]], key=ORDER)
    weak = sorted([update(c) for c in res["weak"]], key=ORDER)
    nearest = update(res["nearest_strong"]) if res["nearest_strong"] else None
    main = pool[:res["k"]]
    changes = [{"name": c["name"], "straight_rank": straight_rank.get(c["unique_id"]), "road_rank": i,
                "road_km": c["road_km"], "estimated": c["road_estimated"]}
               for i, c in enumerate(main, 1) if straight_rank.get(c["unique_id"]) != i]
    shown = {c["unique_id"] for c in main}
    changes += [{"name": c["name"], "straight_rank": straight_rank[c["unique_id"]], "road_rank": None,
                 "road_km": c["road_km"], "estimated": c["road_estimated"]}
                for c in pool if c["unique_id"] in straight_rank and c["unique_id"] not in shown]
    return dict(res, pool=pool, main=main, weak=weak, nearest_strong=nearest,
                road_error=error, road_changes=changes, road_used=bool(roads))


def _adjusted(card: dict, review: dict | None) -> dict:
    """Copy of the card with the recommended rating applied (or unchanged if there is none)."""
    c = dict(card, rule_level=card["evidence_level"], rule_score=card["score"], ai_adjusted=False)
    if not review or review.get("error"):
        return c
    verdict = reconcile(card["evidence_level"], review)
    final = verdict["recommended"]
    if verdict["agree"] or final == card["evidence_level"]:
        return c
    c.update(ai_adjusted=True, evidence_level=final, adjust_reason=verdict["why"])
    if final == "none":
        return c
    parts = card["score_breakdown"]
    new_evidence = LEVEL_SCORE[final]
    c["score_breakdown"] = dict(parts, evidence_score=new_evidence, rule_evidence_score=parts["evidence_score"])
    c["score"] = new_evidence * parts["location_weight"] * parts["distance_factor"]
    return c


def apply_reviews(res: dict, reviews: dict, service: str) -> dict:
    """Return a copy of the search result with ratings, groups and order following the AI suggestion."""
    k = res["k"]
    rule_rank = {c["unique_id"]: i for i, c in enumerate(res["main"], 1)}
    cards = [_adjusted(c, reviews.get((c["unique_id"], service))) for c in res["pool"] + res["weak"]]

    main = sorted([c for c in cards if c["evidence_level"] in ("strong", "moderate")], key=ORDER)[:k]
    weak = sorted([c for c in cards if c["evidence_level"] == "weak"], key=ORDER)[:k]
    removed = [c for c in cards if c["evidence_level"] == "none"]

    nearest = res["nearest_strong"]
    if nearest:
        nearest = _adjusted(nearest, reviews.get((nearest["unique_id"], service)))
        if nearest["evidence_level"] != "strong":
            (removed if nearest["evidence_level"] == "none" else weak).append(nearest)
            nearest = None
    if any(c["evidence_level"] == "strong" for c in main):
        nearest = None  # strong evidence nearby after the review: no separate fallback section

    changes = []
    for i, c in enumerate(main, 1):
        c["final_rank"], c["rule_rank"] = i, rule_rank.get(c["unique_id"])
    for c in main + weak + removed + ([nearest] if nearest else []):
        if c["ai_adjusted"]:
            changes.append({"name": c["name"], "rule": c["rule_level"], "final": c["evidence_level"],
                            "rule_rank": rule_rank.get(c["unique_id"]), "final_rank": c.get("final_rank")})
    return dict(res, main=main, weak=weak, removed=removed, nearest_strong=nearest, changes=changes,
                n_weak=max(res["n_weak"], len(weak)),
                nearby_strong_count=sum(c["evidence_level"] == "strong" for c in main))
