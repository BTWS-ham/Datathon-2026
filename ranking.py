"""Part C: evidence-attached ranking with current and legacy cleaning adapters.

See docs/04_ranking.md for inspected producers and the remaining upstream gap.
The ranking core does not require Spark.
"""

import json
import math
import re
from collections import Counter


FACILITY_TABLE = "workspace.datathon.facilities_clean"
LEGACY_FACILITY_TABLE = "workspace.datathon.ann_facilities_clean_v1"
EVIDENCE_TABLE = "workspace.datathon.facility_service_evidence"
FACILITY_REQUIRED = {
    "facility_id", "name_clean", "city_clean", "state_clean", "latitude",
    "longitude", "flag_missing_coordinates", "flag_invalid_coordinates",
    "flag_outside_india_bbox", "flag_duplicate_id",
    "flag_duplicate_name_city", "flag_duplicate_cluster",
}
EVIDENCE_REQUIRED = {
    "unique_id", "service", "evidence_level", "evidence_score", "quotes",
    "n_direct", "n_supporting",
}
CURRENT_REQUIRED = {
    "unique_id", "name", "city_clean", "state_clean", "latitude", "longitude",
    "location_confidence", "flag_row_corrupted", "flag_coords_missing",
    "flag_coords_outside_india", "flag_coords_far_from_city",
    "flag_pin_state_mismatch", "flag_possible_duplicate", "quality_notes",
}
LOCATION_WEIGHTS = {"high": 0.90, "medium": 0.60, "low": 0.30}
SOURCE_FIELDS = {"description", "procedure", "equipment", "capability", "specialties", "specialty"}
URGENT_SERVICES = {"emergency_c_section", "nicu", "trauma", "burn_care", "emergency", "icu"}
LEVELS = {"strong", "moderate", "weak"}


def normalize_service(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("service must be a non-empty string")
    return re.sub(r"[\s-]+", "_", value.strip().lower())


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance, without road routing or travel-time assumptions."""
    a = (
        math.sin(math.radians(lat2 - lat1) / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def _dict(row):
    return row.asDict(recursive=True) if hasattr(row, "asDict") else dict(row)


def _require(row, keys, label):
    missing = keys - row.keys()
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")


def _adapt_facility(row):
    """Adapt only the two schemas actually observed in teammate code."""
    if "facility_id" in row:
        _require(row, FACILITY_REQUIRED, "Legacy ANN facility")
        return row
    _require(row, CURRENT_REQUIRED, "Current facilities_clean")
    for key in CURRENT_REQUIRED:
        if key.startswith("flag_") and row[key] is not None and type(row[key]) is not bool:
            raise ValueError(f"{key} must be boolean or null")
    return dict(
        row,
        facility_id=row["unique_id"], name_clean=row["name"],
        flag_missing_coordinates=row["flag_coords_missing"],
        flag_invalid_coordinates=False,
        flag_outside_india_bbox=row["flag_coords_outside_india"],
        flag_duplicate_id=False,
        flag_duplicate_name_city=row["flag_possible_duplicate"],
        flag_duplicate_cluster=False,
        _schema="current_facilities_clean",
    )


def _id(value):
    return "" if value is None else str(value).strip()


def _decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _quotes(value):
    """B emits array<struct<field:string,sentence:string>>; preserve its text."""
    value = _decode(value)
    if not isinstance(value, list):
        return []
    result = []
    for quote in value:
        if not isinstance(quote, dict):
            continue
        field = quote.get("field")
        text = quote.get("text", quote.get("sentence"))
        if field in SOURCE_FIELDS and isinstance(text, str) and text.strip():
            result.append({"field": field, "text": text})
    return result


def _source_text(facility, field):
    value = facility.get(field)
    if value is None:
        value = facility.get(f"{field}_items")
    if value is None:
        value = facility.get(f"{field}_list")
    value = _decode(value)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return None if value is None else str(value)


class Ranker:
    def __init__(self, facilities, evidence):
        self.facilities = {}
        self.input_diagnostics = {}
        for raw in facilities:
            row = _adapt_facility(_dict(raw))
            for key in FACILITY_REQUIRED:
                if key.startswith("flag_") and row[key] is not None and type(row[key]) is not bool:
                    raise ValueError(f"{key} must be boolean or null")
            uid = _id(row["facility_id"])
            if not uid:
                continue
            if uid in self.facilities:
                raise ValueError(f"Duplicate facility_id {uid}; fix the upstream join")
            self.facilities[uid] = row

        self.evidence = {}
        seen = set()
        for raw in evidence:
            row = _dict(raw)
            _require(row, EVIDENCE_REQUIRED, "Service evidence")
            service = normalize_service(row["service"])
            uid = _id(row["unique_id"])
            key = (uid, service)
            if key in seen:
                raise ValueError(f"Duplicate facility-service key {key}")
            seen.add(key)
            self.evidence.setdefault(service, []).append(row)

    def recommend(self, service, lat, lon, k=5):
        service = normalize_service(service)
        lat, lon = number(lat), number(lon)
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Query coordinates must be finite WGS84 latitude/longitude")
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError("k must be an integer from 1 to 20")

        urgent = service in URGENT_SERVICES
        nearby_km, decay_km = (30.0, 25.0) if urgent else (60.0, 75.0)
        excluded = Counter()
        candidates = []
        for evidence in self.evidence.get(service, []):
            uid = _id(evidence["unique_id"])
            facility = self.facilities.get(uid)
            if facility is None:
                excluded["missing_facility_match"] += 1
                continue

            if facility.get("flag_row_corrupted") is True:
                excluded["corrupted_facility"] += 1
                continue

            flat, flon = number(facility["latitude"]), number(facility["longitude"])
            # ANN's rectangle is a screening rule, not an India boundary check.
            if (
                flat is None or flon is None
                or not (6 <= flat <= 38 and 68 <= flon <= 98)
                or any(facility[key] is True for key in (
                    "flag_missing_coordinates", "flag_invalid_coordinates",
                    "flag_outside_india_bbox",
                ))
            ):
                excluded["unusable_coordinates"] += 1
                continue
            if facility["flag_duplicate_id"] is True or not facility["name_clean"]:
                excluded["unusable_identity"] += 1
                continue

            score = number(evidence["evidence_score"])
            direct = number(evidence["n_direct"])
            supporting = number(evidence["n_supporting"])
            level = str(evidence["evidence_level"]).strip().lower()
            if (
                score is None or not 0 < score <= 1 or level not in LEVELS
                or direct is None or direct < 0 or not direct.is_integer()
                or supporting is None or supporting < 0 or not supporting.is_integer()
            ):
                excluded["invalid_evidence"] += 1
                continue
            quotes = _quotes(evidence["quotes"])
            if not quotes:
                excluded["missing_structured_quotes"] += 1
                continue
            source_missing = False
            mismatch = False
            for quote in quotes:
                original = _source_text(facility, quote["field"])
                if original is None:
                    source_missing = True
                elif quote["text"] not in original:
                    mismatch = True
            if mismatch:
                excluded["quote_not_in_source"] += 1
                continue

            warnings = ["Facility identity and current service availability are not independently verified."]
            current = facility.get("_schema") == "current_facilities_clean"
            if current:
                location_label = str(facility["location_confidence"]).strip().lower()
                location_weight = LOCATION_WEIGHTS.get(location_label, 0.30)
                weight_source = "facilities_clean.location_confidence_rule_mapping"
                if location_label not in LOCATION_WEIGHTS:
                    warnings.append("Unknown location confidence label; low weight used.")
                if facility.get("quality_notes"):
                    warnings.append(str(facility["quality_notes"]))
                for flag in ("flag_coords_far_from_city", "flag_pin_state_mismatch"):
                    if facility[flag] is True:
                        warnings.append(flag)
                location_statement = (
                    f"The cleaner labels location confidence {location_label}; "
                    "same-city and PIN-prefix checks are heuristic, not independent geographic verification."
                )
                if facility["flag_row_corrupted"] is None:
                    excluded["unknown_corruption_status"] += 1
                    continue
                if any(facility[key] is None for key in CURRENT_REQUIRED if key.startswith("flag_")):
                    location_weight = min(location_weight, 0.30)
                    warnings.append("Some current cleaner quality flags are unknown.")
            else:
                location_label = "unverified"
                location_weight = 0.60
                weight_source = "provisional_ann_v1_flags_proxy"
                location_statement = "The location weight is provisional; city/PIN consistency remains unverified in the legacy ANN cleaner."
                warnings.append(location_statement)
            if any(facility[key] is None for key in FACILITY_REQUIRED if key.startswith("flag_")):
                location_weight = min(location_weight, 0.40)
                warnings.append("Some upstream quality flags are unknown.")
            duplicates = [key for key in ("flag_duplicate_name_city", "flag_duplicate_cluster") if facility[key] is True]
            if duplicates:
                location_weight = min(location_weight, 0.40)
                warnings.append("Possible duplicate or merged identity: " + ", ".join(duplicates))
            if source_missing:
                warnings.append("Some quoted source fields were unavailable for text verification.")
            if level == "strong" and (
                direct < 1 or all(quote["field"] in {"specialties", "specialty"} for quote in quotes)
            ):
                level = "weak"
                warnings.append("Strong label downgraded: direct service-text evidence is missing.")

            distance = distance_km(lat, lon, flat, flon)
            distance_factor = math.exp(-distance / decay_km)
            final_score = score * location_weight * distance_factor
            confidence = (
                "moderate" if level in {"strong", "moderate"}
                and location_weight >= 0.60 and not source_missing else "low"
            )
            candidates.append({
                "unique_id": uid,
                "name": facility["name_clean"],
                "city": facility["city_clean"],
                "state": facility["state_clean"],
                "latitude": flat, "longitude": flon,
                "service": service, "distance_km": distance,
                "score": final_score, "evidence_level": level,
                "location_confidence": location_label,
                "original_evidence_level": evidence["evidence_level"],
                "confidence": confidence,
                "score_breakdown": {
                    "evidence_score": score,
                    "location_weight": location_weight,
                    "location_weight_source": weight_source,
                    "distance_factor": distance_factor,
                    "distance_decay_km": decay_km,
                },
                "n_direct": int(direct), "n_supporting": int(supporting),
                "quotes": quotes, "source_urls": facility.get("source_urls"),
                "data_warnings": warnings,
                "explanation": (
                    f"The dataset provides {level} textual evidence for "
                    f"{service.replace('_', ' ')} at {facility['name_clean']}. "
                    f"The cited fields are {', '.join(sorted({q['field'] for q in quotes}))}. "
                    f"It is approximately {distance:.1f} km from the query point "
                    f"in a straight line. Its ranking score is {score:.3f} "
                    f"× {location_weight:.2f} × {distance_factor:.4f} "
                    f"= {final_score:.4f}. {location_statement} "
                    "Confirm identity, the specific service and availability using "
                    "an independently checked official contact before referral."
                ),
            })

        by_score = lambda item: (-item["score"], item["distance_km"], item["unique_id"])
        nearby = sorted([c for c in candidates if c["distance_km"] <= nearby_km], key=by_score)
        strong = [c for c in candidates if c["evidence_level"] == "strong"]
        nearest_strong = min(strong, key=lambda c: (c["distance_km"], -c["score"], c["unique_id"]), default=None)
        strong_nearby = [c for c in nearby if c["evidence_level"] == "strong"]
        if service not in self.evidence:
            status, message = "unknown_service", "No evidence rows match this service. Check the service vocabulary."
        elif not candidates:
            status, message = "no_usable_candidates", "No records passed the checks. This does not establish a real-world care gap."
        elif strong_nearby:
            status, message = "nearby_strong_evidence", "Nearby strong textual evidence exists; current availability is unverified."
        elif nearest_strong:
            status, message = "no_nearby_strong_evidence", "No nearby strong textual evidence was found. Compare the separate nearest strong option with nearby lower-evidence options."
        else:
            status, message = "no_usable_strong_evidence", "No usable strong textual evidence was found in this dataset; any nearby options need additional verification."

        def select(items, reason):
            return [dict(
                item,
                selection_reason=reason,
                explanation=item["explanation"] + " " + reason,
            ) for item in items]

        nearest_selected = select(
            [nearest_strong] if nearest_strong else [],
            "Selected as the closest usable strong-evidence record by straight-line distance; this is not a travel-time estimate.",
        )
        return {
            "service": service, "status": status, "message": message,
            "query_location": {"latitude": lat, "longitude": lon},
            "policy": {
                "nearby_km": nearby_km, "decay_km": decay_km,
                "distance_method": "great_circle_not_road_or_travel_time",
                "formula": "evidence_score * location_weight * exp(-distance_km / decay_km)",
                "parameters_are_provisional": True,
            },
            "recommendations": select(nearby[:k], "Selected among nearby records by descending combined score."),
            "nearest_strong_option": nearest_selected[0] if nearest_selected else None,
            "nearby_weak_options": select(
                [c for c in nearby if c["evidence_level"] == "weak"][:k],
                "Selected among nearby weak-evidence records by combined score; extra verification is required.",
            ),
            "nearby_lower_evidence_options": select(
                [c for c in nearby if c["evidence_level"] != "strong"][:k],
                "Selected among nearby lower-evidence records by combined score; extra verification is required.",
            ),
            "nearby_strong_count": len(strong_nearby),
            "eligible_candidate_count": len(candidates),
            "excluded_rows": dict(excluded),
            "input_diagnostics": self.input_diagnostics,
        }


def load_from_spark(spark, facility_table=FACILITY_TABLE, evidence_table=EVIDENCE_TABLE, max_rows=200000):
    """Read small prototype tables without mutating them; fail on schema drift."""
    facilities = spark.table(facility_table)
    evidence = spark.table(evidence_table)
    required_facility = FACILITY_REQUIRED if "facility_id" in facilities.columns else CURRENT_REQUIRED
    for frame, required, label in (
        (facilities, required_facility, facility_table),
        (evidence, EVIDENCE_REQUIRED, evidence_table),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
    optional = SOURCE_FIELDS | {f"{field}_items" for field in SOURCE_FIELDS} | {f"{field}_list" for field in SOURCE_FIELDS} | {"source_urls"}
    frows = facilities.select(*sorted(required_facility | (optional & set(facilities.columns)))).limit(max_rows + 1).collect()
    erows = evidence.select(*sorted(EVIDENCE_REQUIRED)).limit(max_rows + 1).collect()
    if len(frows) > max_rows or len(erows) > max_rows:
        raise ValueError("Prototype row limit exceeded; do not silently truncate ranking input")
    # An upstream ID collision cannot be resolved safely by choosing one row.
    # Exclude every facility bearing that ID, including its service evidence.
    counts = Counter(_id(_dict(row).get("facility_id", _dict(row).get("unique_id"))) for row in frows)
    ambiguous_ids = {uid for uid, count in counts.items() if uid and count > 1}
    unambiguous = [
        row for row in frows
        if _id(_dict(row).get("facility_id", _dict(row).get("unique_id"))) not in ambiguous_ids
    ]
    ranker = Ranker(unambiguous, erows)
    ranker.input_diagnostics = {
        "ambiguous_facility_ids_excluded": len(ambiguous_ids),
        "facility_rows_excluded_for_ambiguous_ids": sum(counts[uid] for uid in ambiguous_ids),
    }
    return ranker


_ranker = None


def configure(ranker):
    """Explicitly replace the in-memory snapshot after upstream table refreshes."""
    if not isinstance(ranker, Ranker):
        raise TypeError("Expected Ranker")
    global _ranker
    _ranker = ranker


def recommend(service, lat, lon, k=5):
    if _ranker is None:
        raise RuntimeError("Call configure(load_from_spark(spark)) or configure(Ranker(...)) first")
    return _ranker.recommend(service, lat, lon, k)
