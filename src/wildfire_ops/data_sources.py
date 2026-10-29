"""Data layer: zones, live REST clients (NWS + NASA FIRMS), and scenario replay.

Set WILDFIRE_SCENARIO=path/to/scenario.json to replay recorded conditions
instead of calling live APIs (used by evals and offline demos).
"""
import csv
import io
import json
import math
import os
import re
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
ZONES_FILE = ROOT / "data" / "zones.json"

NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {"User-Agent": "wildfire-ops-agent (github.com/stanislashenon-hub)", "Accept": "application/geo+json"}
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


# ---------- zones & crews ----------

def load_region() -> dict:
    return json.loads(ZONES_FILE.read_text())


def get_zone(zone_id: str) -> dict:
    for z in load_region()["zones"]:
        if z["id"] == zone_id:
            return z
    raise ValueError(f"Unknown zone '{zone_id}'. Call list_zones first.")


# ---------- scenario replay ----------

def _scenario() -> dict | None:
    path = os.getenv("WILDFIRE_SCENARIO")
    return json.loads(Path(path).read_text()) if path else None


# ---------- weather (NWS) ----------

def _max_mph(text: str | None) -> int:
    nums = [int(n) for n in re.findall(r"\d+", text or "")]
    return max(nums) if nums else 0


def get_weather(zone_id: str, hours: int = 24) -> dict:
    """Fire-relevant weather for a zone: active alerts, peak wind, minimum humidity."""
    scen = _scenario()
    if scen:
        return scen["zones"].get(zone_id, {}).get("weather", {"alerts": [], "max_wind_mph": 0, "min_rh_pct": 60})

    z = get_zone(zone_id)
    point = f"{z['lat']},{z['lon']}"
    with httpx.Client(headers=NWS_HEADERS, timeout=20, follow_redirects=True) as c:
        alerts = c.get(f"{NWS_BASE}/alerts/active", params={"point": point}).json()
        meta = c.get(f"{NWS_BASE}/points/{point}").json()
        hourly = c.get(meta["properties"]["forecastHourly"]).json()

    periods = hourly["properties"]["periods"][:hours]
    rh = [p["relativeHumidity"]["value"] for p in periods if p.get("relativeHumidity", {}).get("value") is not None]
    return {
        "alerts": [f["properties"]["event"] for f in alerts.get("features", [])],
        "max_wind_mph": max((_max_mph(p.get("windSpeed")) for p in periods), default=0),
        "min_rh_pct": min(rh) if rh else None,
    }


# ---------- active fires (NASA FIRMS) ----------

def _km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def get_fires(zone_id: str, radius_km: float = 25, days: int = 1) -> list[dict]:
    """Satellite fire detections within radius_km of the zone, nearest first."""
    scen = _scenario()
    if scen:
        fires = scen["zones"].get(zone_id, {}).get("fires", [])
        return sorted([f for f in fires if f["distance_km"] <= radius_km], key=lambda f: f["distance_km"])

    key = os.getenv("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("FIRMS_MAP_KEY not set (free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/)")
    z = get_zone(zone_id)
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(z["lat"])))
    bbox = f"{z['lon'] - dlon:.4f},{z['lat'] - dlat:.4f},{z['lon'] + dlon:.4f},{z['lat'] + dlat:.4f}"
    resp = httpx.get(f"{FIRMS_BASE}/{key}/VIIRS_SNPP_NRT/{bbox}/{days}", timeout=30)
    resp.raise_for_status()

    fires = []
    for row in csv.DictReader(io.StringIO(resp.text)):
        d = _km(z["lat"], z["lon"], float(row["latitude"]), float(row["longitude"]))
        if d <= radius_km:
            fires.append({
                "distance_km": round(d, 1),
                "frp_mw": float(row.get("frp") or 0),
                "confidence": row.get("confidence"),
                "detected": f"{row.get('acq_date')} {row.get('acq_time')}",
            })
    return sorted(fires, key=lambda f: f["distance_km"])
