"""Offline tests: no API keys, no network. Run: pytest -q"""
import asyncio
import json
import os
from pathlib import Path

import pytest
from langgraph.types import Command

from wildfire_ops import data_sources as ds
from wildfire_ops.agent import build_graph, mcp_client
from wildfire_ops.risk import score_zone

SCEN = Path(__file__).resolve().parents[1] / "data" / "scenarios"


@pytest.mark.parametrize("path", sorted(SCEN.glob("*.json")), ids=lambda p: p.stem)
def test_scores_match_expected_labels(path, monkeypatch):
    """The deterministic scorer agrees with each scenario's expected levels."""
    monkeypatch.setenv("WILDFIRE_SCENARIO", str(path))
    scen = json.loads(path.read_text())
    for zone_id, level in scen["expected"]["levels"].items():
        got = score_zone(ds.get_weather(zone_id), ds.get_fires(zone_id))["recommended_level"]
        assert got == level, f"{zone_id}: {got} != {level}"


def test_weather_alone_never_evacuates():
    r = score_zone({"alerts": ["Red Flag Warning"], "max_wind_mph": 60, "min_rh_pct": 3}, [])
    assert r["recommended_level"] == "advisory"


def test_mcp_server_end_to_end(monkeypatch, tmp_path):
    """Spawn the real MCP server over stdio and call tools through the protocol."""
    monkeypatch.setenv("WILDFIRE_SCENARIO", str(SCEN / "oakland_hills_fire.json"))

    async def go():
        tools = {t.name: t for t in await mcp_client().get_tools()}
        assert {"list_zones", "assess_zone_risk", "issue_warning", "dispatch_crew"} <= set(tools)
        out = await tools["assess_zone_risk"].ainvoke({"zone_id": "OAK-HILLS"})
        return str(out)

    assert "evacuation_order" in asyncio.run(go())


def test_graph_human_in_the_loop(monkeypatch):
    """Only approved actions reach the action tools."""
    monkeypatch.setenv("WILDFIRE_SCENARIO", str(SCEN / "oakland_hills_fire.json"))
    calls = []

    class FakeTool:
        def __init__(self, name): self.name = name
        async def ainvoke(self, args): calls.append((self.name, args)); return "ok"

    async def fake_investigate(state):
        return {"situation_summary": "test", "actions": [
            {"id": 1, "type": "issue_warning", "zone_id": "OAK-HILLS", "level": "evacuation_order",
             "crew_id": None, "text": "Leave now", "evidence": "score 100"},
            {"id": 2, "type": "dispatch_crew", "zone_id": "OAK-HILLS", "level": None,
             "crew_id": "ENG-21", "text": "Structure defense", "evidence": "fire 3 km"}]}

    async def go():
        g = build_graph([FakeTool("issue_warning"), FakeTool("dispatch_crew")], investigate_fn=fake_investigate)
        cfg = {"configurable": {"thread_id": "t1"}}
        s = await g.ainvoke({"question": "x"}, cfg)
        assert "__interrupt__" in s and not calls  # paused, nothing executed yet
        return await g.ainvoke(Command(resume=[2]), cfg)

    final = asyncio.run(go())
    assert calls == [("dispatch_crew", {"crew_id": "ENG-21", "zone_id": "OAK-HILLS", "reason": "Structure defense"})]
    assert final["report"].startswith("2 actions proposed, 1 approved, 1 executed")


def test_nws_wind_parsing():
    assert ds._max_mph("10 to 25 mph") == 25
    assert ds._max_mph("15 mph") == 15
    assert ds._max_mph(None) == 0
