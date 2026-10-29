"""MCP server: exposes the wildfire 'ontology' (zones, crews) and its tools.

Read tools gather facts. Action tools change the world (simulated here:
they append to data/actions_log.jsonl). The LangGraph agent only calls
action tools after a human approves.

Run:  python -m wildfire_ops.mcp_server        (stdio, for Claude Desktop / the agent)
"""
import json
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from wildfire_ops import data_sources as ds
from wildfire_ops.risk import score_zone

ACTIONS_LOG = ds.ROOT / "data" / "actions_log.jsonl"

mcp = FastMCP("wildfire-ops", log_level="WARNING")


# ---------- resources ----------

@mcp.resource("region://zones")
def zones_resource() -> str:
    """All monitored zones and available crews."""
    return json.dumps(ds.load_region(), indent=2)


# ---------- read tools ----------

@mcp.tool()
def list_zones() -> dict:
    """List monitored zones (id, name, population, notes) and available crews."""
    return ds.load_region()


@mcp.tool()
def get_weather_risk(zone_id: str) -> dict:
    """Next-24h fire weather for a zone: NWS alerts, peak wind (mph), minimum humidity (%)."""
    return ds.get_weather(zone_id)


@mcp.tool()
def get_active_fires(zone_id: str, radius_km: float = 25) -> list[dict]:
    """NASA FIRMS satellite fire detections within radius_km of a zone, nearest first."""
    return ds.get_fires(zone_id, radius_km)


@mcp.tool()
def assess_zone_risk(zone_id: str) -> dict:
    """Combine weather + fires into a 0-100 risk score with reasons and a recommended alert level."""
    zone = ds.get_zone(zone_id)
    weather = ds.get_weather(zone_id)
    fires = ds.get_fires(zone_id)
    return {"zone_id": zone_id, "zone_name": zone["name"], "population": zone["population"],
            **score_zone(weather, fires), "weather": weather, "nearest_fires": fires[:3]}


# ---------- action tools (human-approved only) ----------

def _log(action: dict) -> dict:
    action["timestamp"] = datetime.now(timezone.utc).isoformat()
    with ACTIONS_LOG.open("a") as f:
        f.write(json.dumps(action) + "\n")
    return {"status": "logged (simulated)", **action}


@mcp.tool()
def issue_warning(zone_id: str, level: str, message: str) -> dict:
    """Issue a public alert for a zone. level: advisory | evacuation_warning | evacuation_order."""
    ds.get_zone(zone_id)
    return _log({"type": "issue_warning", "zone_id": zone_id, "level": level, "message": message})


@mcp.tool()
def dispatch_crew(crew_id: str, zone_id: str, reason: str) -> dict:
    """Send a crew to a zone."""
    ds.get_zone(zone_id)
    if crew_id not in {c["id"] for c in ds.load_region()["crews"]}:
        raise ValueError(f"Unknown crew '{crew_id}'")
    return _log({"type": "dispatch_crew", "crew_id": crew_id, "zone_id": zone_id, "reason": reason})


if __name__ == "__main__":
    mcp.run()
