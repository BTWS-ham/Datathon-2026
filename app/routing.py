"""Road distances from the search point to candidate facilities, via OSRM (OpenStreetMap routing).

One request to the OSRM "table" service returns the driving distance from the search point to
every candidate. No live traffic is used. If the service cannot be reached, callers fall back to
an estimated road distance (ROAD_FACTOR x straight line), labelled as an estimate.

The public demo server (router.project-osrm.org) is meant for light use. For production, point
OSRM_URL at a self-hosted OSRM. Only coordinates are sent: the search point and facility locations.
"""

from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache

OSRM_URL = os.getenv("OSRM_URL", "https://router.project-osrm.org")
TIMEOUT_S = float(os.getenv("OSRM_TIMEOUT", "6"))
MAX_DESTINATIONS = 90  # keep the request URL short
SOURCE_NAME = "OpenStreetMap road network (OSRM)"
# Fallback only: typical ratio of road to straight-line distance (road networks are roughly
# 1.3-1.5x longer). Used when OSRM is unreachable or has no route; always labelled "estimated".
ROAD_FACTOR = 1.4


@lru_cache(maxsize=512)
def _table(origin: tuple[float, float], dests: tuple[tuple[float, float], ...]) -> tuple:
    coords = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in (origin, *dests))
    url = f"{OSRM_URL}/table/v1/driving/{coords}?sources=0&annotations=distance"
    req = urllib.request.Request(url, headers={"User-Agent": "carefinder-datathon-app"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.load(resp)
    if data.get("code") != "Ok":
        raise RuntimeError(data.get("message") or data.get("code"))
    return tuple(data["distances"][0][1:])  # metres, None where no road route exists


def _stub(origin, points):
    """APP_ROUTING=stub (tests): road = 1.3 x straight line, plus 40 km for ids containing 'detour'."""
    from math import asin, cos, radians, sin, sqrt
    out = {}
    for uid, (lat, lon) in points.items():
        a = sin(radians(lat - origin[0]) / 2) ** 2 + cos(radians(origin[0])) * cos(radians(lat)) * sin(radians(lon - origin[1]) / 2) ** 2
        km = 2 * 6371.0088 * asin(sqrt(min(1.0, a)))
        out[uid] = km * 1.3 + (40.0 if "detour" in uid else 0.0)
    return out


def road_km(origin: tuple[float, float], points: dict[str, tuple[float, float]]) -> tuple[dict, str | None]:
    """Return ({facility id: road km or None}, error message or None)."""
    mode = os.getenv("APP_ROUTING", "on").lower()
    if mode == "off":
        return {}, "road distances are turned off (APP_ROUTING=off)"
    if mode == "stub":
        return _stub(origin, points), None
    if not points:
        return {}, None
    ids = list(points)
    out = {}
    try:
        for start in range(0, len(ids), MAX_DESTINATIONS):
            chunk = ids[start:start + MAX_DESTINATIONS]
            dests = tuple((round(points[i][0], 5), round(points[i][1], 5)) for i in chunk)
            metres = _table((round(origin[0], 5), round(origin[1], 5)), dests)
            out.update({i: (m / 1000 if m is not None else None) for i, m in zip(chunk, metres)})
    except Exception as e:  # network, rate limit or service error: caller shows straight-line instead
        return {}, f"road distances unavailable ({type(e).__name__})"
    return out, None
