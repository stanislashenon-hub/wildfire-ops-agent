"""Replay every scenario, auto-approve the agent's proposal, and score it.

  python evals/run_evals.py

Metrics per scenario:
  level_accuracy   zones where the issued alert level == expected level
  missed_critical  zones expected at evacuation_warning+ that got less (the metric that matters most)
  false_alarms     zones expected at 'monitor' that got an alert
  crew_ok          a crew was dispatched to every zone in expected.crew_to
"""
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildfire_ops.agent import build_graph, flush_langfuse, langfuse_callbacks, mcp_client  # noqa: E402

RANK = {"monitor": 0, "advisory": 1, "evacuation_warning": 2, "evacuation_order": 3}


def score(proposal: dict, expected: dict) -> dict:
    issued = {}
    for a in proposal["actions"]:
        if a["type"] == "issue_warning":
            z = a["zone_id"]
            issued[z] = max(issued.get(z, "monitor"), a["level"], key=RANK.get)
    crews_to = {a["zone_id"] for a in proposal["actions"] if a["type"] == "dispatch_crew"}

    levels = expected["levels"]
    got = {z: issued.get(z, "monitor") for z in levels}
    return {
        "level_accuracy": sum(got[z] == levels[z] for z in levels) / len(levels),
        "missed_critical": [z for z in levels if RANK[levels[z]] >= 2 and RANK[got[z]] < RANK[levels[z]]],
        "false_alarms": [z for z in levels if levels[z] == "monitor" and got[z] != "monitor"],
        "crew_ok": set(expected["crew_to"]) <= crews_to,
        "got": got,
    }


async def run_scenario(path: Path) -> dict:
    os.environ["WILDFIRE_SCENARIO"] = str(path)  # inherited by the MCP server subprocess
    scen = json.loads(path.read_text())
    tools = await mcp_client().get_tools()
    graph = build_graph(tools)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": langfuse_callbacks(),
              "metadata": {"scenario": path.stem}}
    state = await graph.ainvoke({"question": "Assess all zones. Which need alerts or crews right now?"}, config)
    proposal = state["__interrupt__"][0].value
    await graph.ainvoke(Command(resume="none"), config)  # evals never execute actions
    return {"scenario": path.stem, **score(proposal, scen["expected"])}


async def main():
    results = []
    for path in sorted((ROOT / "data" / "scenarios").glob("*.json")):
        r = await run_scenario(path)
        results.append(r)
        print(f"{r['scenario']:<24} acc={r['level_accuracy']:.0%}  missed_critical={r['missed_critical']}  "
              f"false_alarms={r['false_alarms']}  crew_ok={r['crew_ok']}")
    n = len(results)
    print(f"\nOVERALL  acc={sum(r['level_accuracy'] for r in results) / n:.0%}  "
          f"missed_critical={sum(len(r['missed_critical']) for r in results)}  "
          f"crew_ok={sum(r['crew_ok'] for r in results)}/{n}")
    (ROOT / "evals" / "results.json").write_text(json.dumps(results, indent=2))
    flush_langfuse()


if __name__ == "__main__":
    load_dotenv(ROOT / ".env", override=True)
    asyncio.run(main())
