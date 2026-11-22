# Snapshot from 01_clean_facilities.ipynb, cell 3, commit b89a5f5ad20f1eae14a0039f3393b6375e7c9a20.
# Team upstream logic preserved verbatim for reproducible Part C integration.

# ===== Cleaning logic (pure pandas; 10k rows fits easily on the driver) =====
import json, re, difflib
import numpy as np
import pandas as pd

# ---------- Canonical states / union territories ----------
STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]
STATE_ALIASES = {
    "up": "Uttar Pradesh", "u.p.": "Uttar Pradesh", "u.p": "Uttar Pradesh",
    "uttaranchal": "Uttarakhand", "uttranchal": "Uttarakhand",
    "tamilnadu": "Tamil Nadu", "orissa": "Odisha", "pondicherry": "Puducherry",
    "new delhi": "Delhi", "nct of delhi": "Delhi", "delhi ncr": "Delhi",
    "chandigarh (ut)": "Chandigarh", "j&k": "Jammu and Kashmir", "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir", "mp": "Madhya Pradesh", "ap": "Andhra Pradesh",
    "wb": "West Bengal", "daman": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "andaman and nicobar": "Andaman and Nicobar Islands",
    # common city names that appear in the state field
    "hyderabad": "Telangana", "mumbai": "Maharashtra", "bangalore": "Karnataka",
    "bengaluru": "Karnataka", "kolkata": "West Bengal", "chennai": "Tamil Nadu",
}
_STATE_LOWER = {s.lower(): s for s in STATES}

# ---------- PIN prefix -> plausible states (heuristic) ----------
PIN2 = {
    "11": ["Delhi"], "12": ["Haryana"], "13": ["Haryana"], "14": ["Punjab"], "15": ["Punjab"],
    "16": ["Punjab", "Chandigarh", "Haryana"], "17": ["Himachal Pradesh"],
    "18": ["Jammu and Kashmir"], "19": ["Jammu and Kashmir", "Ladakh"],
    "20": ["Uttar Pradesh"], "21": ["Uttar Pradesh"], "22": ["Uttar Pradesh"], "23": ["Uttar Pradesh"],
    "24": ["Uttar Pradesh", "Uttarakhand"], "25": ["Uttar Pradesh"], "26": ["Uttar Pradesh", "Uttarakhand"],
    "27": ["Uttar Pradesh"], "28": ["Uttar Pradesh"],
    "30": ["Rajasthan"], "31": ["Rajasthan"], "32": ["Rajasthan"], "33": ["Rajasthan"], "34": ["Rajasthan"],
    "36": ["Gujarat"], "37": ["Gujarat"], "38": ["Gujarat"],
    "39": ["Gujarat", "Dadra and Nagar Haveli and Daman and Diu"],
    "40": ["Maharashtra", "Goa"], "41": ["Maharashtra"], "42": ["Maharashtra"], "43": ["Maharashtra"],
    "44": ["Maharashtra"], "45": ["Madhya Pradesh"], "46": ["Madhya Pradesh"], "47": ["Madhya Pradesh"],
    "48": ["Madhya Pradesh"], "49": ["Chhattisgarh"],
    "50": ["Telangana"], "51": ["Andhra Pradesh", "Telangana"], "52": ["Andhra Pradesh"],
    "53": ["Andhra Pradesh"], "56": ["Karnataka"], "57": ["Karnataka"], "58": ["Karnataka"],
    "59": ["Karnataka"], "60": ["Tamil Nadu", "Puducherry"], "61": ["Tamil Nadu"], "62": ["Tamil Nadu"],
    "63": ["Tamil Nadu"], "64": ["Tamil Nadu"], "67": ["Kerala"], "68": ["Kerala", "Lakshadweep"],
    "69": ["Kerala"], "70": ["West Bengal"], "71": ["West Bengal"], "72": ["West Bengal"],
    "73": ["West Bengal", "Sikkim"], "74": ["West Bengal", "Andaman and Nicobar Islands"],
    "75": ["Odisha"], "76": ["Odisha"], "77": ["Odisha"], "78": ["Assam"],
    "79": ["Arunachal Pradesh", "Meghalaya", "Manipur", "Mizoram", "Nagaland", "Tripura", "Assam"],
    "80": ["Bihar"], "81": ["Bihar", "Jharkhand"], "82": ["Bihar", "Jharkhand"], "83": ["Jharkhand"],
    "84": ["Bihar"], "85": ["Bihar"],
}

# Rough India bounding box (only catches grossly wrong coordinates)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 6.0, 37.5, 68.0, 97.5
FAR_KM, VERY_FAR_KM, MIN_CITY_N = 50, 150, 3
KNOWN_FACILITY_TYPES = {"hospital", "clinic", "dentist", "doctor", "pharmacy", "farmacy"}
KNOWN_OPERATOR_TYPES = {"private", "public", "government"}
EVIDENCE_FIELDS = ["specialties", "procedure", "equipment", "capability"]


def is_missing(x):
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    return isinstance(x, str) and x.strip().lower() in ("", "null", "none", "n/a", "na")


def parse_list(x):
    """Parse an evidence field. Returns (clean list, parse_failed).

    Non-JSON values in these columns are shifted-in junk (coordinates, `true`, IDs),
    not evidence text, so a failed parse gives [] and is flagged as misaligned.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        v = list(x)
    elif is_missing(x):
        return [], False
    else:
        try:
            v = json.loads(str(x))
        except Exception:
            return [], True
        if not isinstance(v, list):
            return [], True
    return [s.strip() for s in v if isinstance(s, str) and s.strip()], False


def dedup_keep_order(items):
    seen, out = set(), []
    for s in items:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def norm_text(s):
    if is_missing(s):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def normalize_state(raw):
    """Map a raw state string to a canonical state. Returns (state or None, method)."""
    if is_missing(raw):
        return None, "missing"
    s = re.sub(r"\s+", " ", str(raw).strip().lower())
    if s in _STATE_LOWER:
        return _STATE_LOWER[s], "exact"
    s2 = s.replace("&", " and ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    if s2 in _STATE_LOWER:
        return _STATE_LOWER[s2], "exact"
    if s in STATE_ALIASES:
        return STATE_ALIASES[s], "alias"
    if s2 in STATE_ALIASES:
        return STATE_ALIASES[s2], "alias"
    m = difflib.get_close_matches(s2, list(_STATE_LOWER), n=1, cutoff=0.85)
    if m:
        return _STATE_LOWER[m[0]], "fuzzy"
    return None, "unrecognized"


def clean_pin(x):
    if is_missing(x):
        return None
    s = re.sub(r"\.0$", "", str(x).strip()).replace(" ", "")
    return s if re.fullmatch(r"[1-9]\d{5}", s) else None


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def clean_facilities(df: pd.DataFrame) -> pd.DataFrame:
    # Positional RangeIndex: masks and the `notes` list below are aligned by position
    d = df.reset_index(drop=True).copy()
    notes = [[] for _ in range(len(d))]

    def note(mask, text):
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            notes[i].append(text)

    # ---- Numeric fields (safe cast; junk -> NaN) ----
    for c in ["latitude", "longitude", "numberDoctors", "capacity"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["year_established"] = pd.to_numeric(d["yearEstablished"], errors="coerce")
    d.loc[~d["year_established"].between(1800, 2026), "year_established"] = np.nan

    # ---- Evidence lists: parse + deduplicate ----
    list_parse_failed = pd.Series(False, index=d.index)
    for c in EVIDENCE_FIELDS:
        parsed = d[c].map(parse_list)
        d[c + "_list"] = parsed.map(lambda t: dedup_keep_order(t[0]))
        d["n_" + c] = d[c + "_list"].map(len)
        list_parse_failed |= parsed.map(lambda t: t[1])

    # ---- Misaligned fields (extraction shifted columns) ----
    ft = d["facilityTypeId"].map(lambda v: "" if is_missing(v) else str(v).strip().lower())
    op = d["operatorTypeId"].map(lambda v: "" if is_missing(v) else str(v).strip().lower())
    d["facility_type"] = ft.replace({"farmacy": "pharmacy"}).where(ft.isin(KNOWN_FACILITY_TYPES), None)
    d["operator_type"] = op.replace({"government": "public"}).where(op.isin(KNOWN_OPERATOR_TYPES), None)
    misaligned = (
        ((ft != "") & ~ft.isin(KNOWN_FACILITY_TYPES))
        | ((op != "") & ~op.isin(KNOWN_OPERATOR_TYPES))
        | list_parse_failed
    )
    d["flag_field_misaligned"] = misaligned
    note(misaligned, "Some fields look shifted/corrupted in extraction")

    # ---- PIN code ----
    d["pin_code"] = d["address_zipOrPostcode"].map(clean_pin)
    d["pin_states"] = d["pin_code"].map(lambda p: PIN2.get(p[:2], []) if isinstance(p, str) else [])
    note(d["pin_code"].isna(), "PIN code missing or invalid")

    # ---- State normalization (given state first, PIN fallback) ----
    ns = d["address_stateOrRegion"].map(normalize_state)
    d["state_given"] = ns.map(lambda t: t[0])
    d["state_method"] = ns.map(lambda t: t[1])
    # Pre-2014 records list Telangana cities (PIN 50xxxx) as Andhra Pradesh: treat as Telangana
    ap_tg = (d["state_given"] == "Andhra Pradesh") & d["pin_code"].fillna("").str.startswith("50")
    d.loc[ap_tg, "state_given"] = "Telangana"
    d.loc[ap_tg, "state_method"] = "corrected_ap_to_telangana"
    pin_single = d["pin_states"].map(lambda s: s[0] if len(s) == 1 else None)
    infer = d["state_given"].isna() & pin_single.notna()
    d["state_clean"] = d["state_given"].where(d["state_given"].notna(), pin_single)
    d.loc[infer, "state_method"] = "inferred_from_pin"
    mism = pd.Series(
        [pd.notna(g) and len(ps) > 0 and g not in ps
         for g, ps in zip(d["state_given"], d["pin_states"])],
        index=d.index)
    d["flag_pin_state_mismatch"] = mism
    note(mism, "PIN code does not match the stated state")

    # ---- City ----
    city = d["address_city"].map(lambda v: "" if is_missing(v) else re.sub(r"\s+", " ", str(v).strip()).title())
    d["city_clean"] = city.mask(city == "")
    d["city_key"] = city.map(norm_text)

    # ---- Coordinates ----
    has = d["latitude"].notna() & d["longitude"].notna()
    in_india = has & d["latitude"].between(LAT_MIN, LAT_MAX) & d["longitude"].between(LON_MIN, LON_MAX)
    d["flag_coords_missing"] = ~has
    d["flag_coords_outside_india"] = has & ~in_india
    note(~has, "Coordinates missing")
    note(has & ~in_india, "Coordinates fall outside India")

    # Distance to the median location of facilities in the same state + city (robust city centre).
    # Only for cities with >= MIN_CITY_N facilities with in-India coordinates.
    grp_key = d["state_clean"].fillna("?") + "|" + d["city_key"]
    usable = in_india & (d["city_key"] != "")
    cen = (d[usable].assign(k=grp_key[usable])
           .groupby("k").agg(c_lat=("latitude", "median"), c_lon=("longitude", "median"),
                             city_n=("latitude", "size")))
    c_lat = grp_key.map(cen["c_lat"])
    c_lon = grp_key.map(cen["c_lon"])
    city_n = grp_key.map(cen["city_n"]).fillna(0)
    dist = haversine_km(d["latitude"], d["longitude"], c_lat, c_lon)
    d["km_from_city_centre"] = np.where(usable & (city_n >= MIN_CITY_N), np.round(dist, 1), np.nan)
    far = d["km_from_city_centre"] > FAR_KM
    very_far = d["km_from_city_centre"] > VERY_FAR_KM
    d["flag_coords_far_from_city"] = far
    note(far & ~very_far, f"Coordinates {FAR_KM}-{VERY_FAR_KM} km from other facilities in the same city")
    note(very_far, f"Coordinates >{VERY_FAR_KM} km from other facilities in the same city")

    # ---- Shared phone numbers (aggregator / call-centre numbers) ----
    ph = d["officialPhone"].map(lambda v: "" if is_missing(v) else re.sub(r"\D", "", str(v)))
    name_key = d["name"].map(norm_text)
    tmp = pd.DataFrame({"ph": ph, "nm": name_key})
    shared = tmp[(tmp.ph.str.len() >= 8) & (tmp.nm != "")].groupby("ph")["nm"].nunique()
    d["phone_shared_by_n"] = ph.map(shared).fillna(0).astype(int)
    d["flag_phone_shared"] = d["phone_shared_by_n"] >= 3
    note(d["flag_phone_shared"], "Official phone is shared with other, unrelated facilities")

    # ---- Duplicates (same normalized name in the same city) ----
    key = name_key + "|" + d["city_key"]
    valid_key = (name_key != "") & (d["city_key"] != "")
    d["dup_count"] = key.map(key[valid_key].value_counts()).where(valid_key, 1).fillna(1).astype(int)
    d["flag_possible_duplicate"] = d["dup_count"] > 1
    note(d["flag_possible_duplicate"], "Another record has the same name in the same city")

    # ---- Capability bloat (small facility claiming many specialties) ----
    bloat = d["facility_type"].isin(["clinic", "dentist", "doctor"]) & (d["n_specialties"] >= 20)
    d["flag_capability_bloat"] = bloat
    note(bloat, "Small facility lists an unusually large number of specialties")

    # ---- Corrupted rows (whole record shifted during extraction; ~90 rows) ----
    # Checked on full data: names starting with a digit ("32 Pearls Dental Clinic"), lowercase
    # ("iScan Imaging Centre") or a quote are real names, so only these patterns count as corrupted.
    raw_name = d["name"].map(lambda v: "" if is_missing(v) else str(v))
    corrupted = (
        misaligned
        | (raw_name == "")                       # name missing (city usually missing too)
        | raw_name.str.match(r"^\s*[\[{]")       # JSON specialty list spilled into the name column
        | raw_name.str.match(r"^\s")             # sentence fragment from the description
    )
    d["flag_row_corrupted"] = corrupted
    note(corrupted, "Record is corrupted (no usable facility name) - exclude from recommendations")

    # ---- Completeness of supporting fields (0-1) ----
    website_ok = d["officialWebsite"].map(lambda v: not is_missing(v))
    comp_cols = [d["numberDoctors"].notna(), d["capacity"].notna(), d["year_established"].notna(),
                 d["n_equipment"] > 0, d["n_procedure"] > 0, website_ok]
    d["completeness"] = (np.column_stack(comp_cols).sum(axis=1) / len(comp_cols)).round(2)

    # ---- Location confidence ----
    low = (~has) | d["flag_coords_outside_india"] | very_far | corrupted
    medium = (~low) & (far | mism | d["state_clean"].isna() | d["km_from_city_centre"].isna())
    d["location_confidence"] = np.select([low, medium], ["low", "medium"], "high")

    d["quality_notes"] = ["; ".join(n) for n in notes]
    d["n_quality_flags"] = [len(n) for n in notes]
    return d


OUTPUT_COLUMNS = [
    "unique_id", "name", "facility_type", "operator_type",
    "address_line1", "address_line2", "city_clean", "state_clean", "state_method", "pin_code",
    "latitude", "longitude", "km_from_city_centre", "location_confidence",
    "numberDoctors", "capacity", "year_established", "completeness",
    "description", "specialties_list", "procedure_list", "equipment_list", "capability_list",
    "n_specialties", "n_procedure", "n_equipment", "n_capability",
    "officialWebsite", "officialPhone", "phone_shared_by_n", "source_urls", "cluster_id", "dup_count",
    "flag_coords_missing", "flag_coords_outside_india", "flag_coords_far_from_city",
    "flag_pin_state_mismatch", "flag_phone_shared", "flag_possible_duplicate",
    "flag_capability_bloat", "flag_field_misaligned", "flag_row_corrupted",
    "n_quality_flags", "quality_notes",
]
