"""Turn what a user types (city, district, PIN code or coordinates) into a query point.

City and PIN centres come from facilities whose own coordinates the cleaner trusts
(location_confidence == "high"); the optional app_locations table adds India Post PIN
codes and districts that have no facilities.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

import pandas as pd

LAT_RANGE, LON_RANGE = (6.0, 37.5), (68.0, 97.5)


def norm_key(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def in_india(lat, lon) -> bool:
    return (lat is not None and lon is not None
            and LAT_RANGE[0] <= lat <= LAT_RANGE[1] and LON_RANGE[0] <= lon <= LON_RANGE[1])


def _num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


@dataclass
class ResolvedLocation:
    lat: float
    lon: float
    label: str
    method: str          # coordinates | pin | city | district
    precision: str       # plain-English note on how exact the point is
    alternatives: list = field(default_factory=list)


class PlaceIndex:
    def __init__(self, facilities: pd.DataFrame, locations: pd.DataFrame | None = None):
        fac = facilities.copy()
        fac["latitude"] = pd.to_numeric(fac["latitude"], errors="coerce")
        fac["longitude"] = pd.to_numeric(fac["longitude"], errors="coerce")
        fac["city_key"] = fac["city_clean"].map(norm_key)
        good = fac[(fac["location_confidence"] == "high") & fac["latitude"].notna() & fac["longitude"].notna()]

        cities = good[good["city_key"] != ""]
        grp = cities.groupby([cities["state_clean"].fillna(""), "city_key"])
        cen = grp.agg(lat=("latitude", "median"), lon=("longitude", "median"),
                      n=("latitude", "size"), city=("city_clean", "first")).reset_index()
        places = [{"kind": "city", "name": r.city, "key": r.city_key, "state": r.state_clean or None,
                   "lat": float(r.lat), "lon": float(r.lon), "n": int(r.n),
                   "source": f"median of {int(r.n)} facility locations"} for r in cen.itertuples()]

        if locations is not None and len(locations):
            for r in locations.to_dict("records"):
                lat, lon = _num(r.get("lat")), _num(r.get("lon"))
                if not in_india(lat, lon):
                    continue
                places.append({"kind": str(r.get("kind")), "name": str(r.get("name")),
                               "key": norm_key(r.get("name")), "state": r.get("state"),
                               "lat": lat, "lon": lon, "n": int(_num(r.get("n_points")) or 1),
                               "source": str(r.get("source") or "India Post PIN directory")})
        self.places = pd.DataFrame(places, columns=["kind", "name", "key", "state", "lat", "lon", "n", "source"])

        pins = good[good["pin_code"].notna()] if "pin_code" in good else good.iloc[0:0]
        self.facility_pins = (pins.groupby("pin_code").agg(lat=("latitude", "median"), lon=("longitude", "median"),
                                                           n=("latitude", "size"))
                              if len(pins) else pd.DataFrame(columns=["lat", "lon", "n"]))

    def resolve(self, text: str) -> ResolvedLocation | None:
        text = (text or "").strip()
        if not text:
            return None

        m = re.fullmatch(r"\(?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\)?", text)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if not in_india(lat, lon) and in_india(lon, lat):
                lat, lon = lon, lat
            if in_india(lat, lon):
                return ResolvedLocation(lat, lon, f"{lat:.4f}, {lon:.4f}", "coordinates",
                                        "Exact coordinates you entered.")
            return None

        compact = text.replace(" ", "")
        if re.fullmatch(r"[1-9]\d{5}", compact):
            p = self.places
            hit = p[(p["kind"] == "pin") & (p["name"] == compact)]
            if len(hit):
                h = hit.iloc[0]
                return ResolvedLocation(float(h["lat"]), float(h["lon"]), f"PIN {compact}", "pin",
                                        f"Centre of PIN area {compact} ({h['source']}); may be several km off.")
            if compact in self.facility_pins.index:
                h = self.facility_pins.loc[compact]
                return ResolvedLocation(float(h["lat"]), float(h["lon"]), f"PIN {compact}", "pin",
                                        f"Approximate centre of PIN {compact}, from {int(h['n'])} facility locations.")
            return None

        key = norm_key(text.split(",")[0])
        state_hint = norm_key(text.split(",")[1]) if "," in text else ""
        p = self.places
        for kind in ("city", "district"):
            hit = p[(p["kind"] == kind) & (p["key"] == key)]
            if state_hint:
                in_state = hit[hit["state"].map(norm_key) == state_hint]
                hit = in_state if len(in_state) else hit
            if len(hit):
                hit = hit.sort_values("n", ascending=False)
                h = hit.iloc[0]
                where = f"{h['name']}, {h['state']}" if h["state"] else str(h["name"])
                alternatives = [f"{a['name']}, {a['state']}" for _, a in hit.iloc[1:4].iterrows()]
                return ResolvedLocation(float(h["lat"]), float(h["lon"]), where, kind,
                                        f"Centre of {where} ({h['source']}). Distances are straight-line from this point.",
                                        alternatives)
        return None

    def suggest(self, text: str, n: int = 5) -> list[str]:
        p = self.places[self.places["kind"].isin(["city", "district"])].sort_values("n", ascending=False)
        uniq = p.drop_duplicates("key")
        out = []
        for k in difflib.get_close_matches(norm_key(text), uniq["key"].tolist(), n=n, cutoff=0.75):
            r = uniq[uniq["key"] == k].iloc[0]
            out.append(f"{r['name']}, {r['state']}" if r["state"] else str(r["name"]))
        return out
