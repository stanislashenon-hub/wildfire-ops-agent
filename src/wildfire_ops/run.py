"""CLI: run the agent with a human approval step in the terminal.

  python -m wildfire_ops.run "Which zones need action tonight?"
  WILDFIRE_SCENARIO=data/scenarios/oakland_hills_fire.json python -m wildfire_ops.run
"""
import argparse
import asyncio
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from wildfire_ops.agent import build_graph, flush_langfuse, langfuse_callbacks, mcp_client

LABEL = {"advisory": "ADVISORY", "evacuation_warning": "EVAC WARNING", "evacuation_order": "EVAC ORDER"}


def show(proposal: dict):
    print("\n=== SITUATION ===\n" + proposal["situation_summary"])
    print("\n=== PROPOSED ACTIONS ===")
    for a in proposal["actions"]:
        what = LABEL.get(a.get("level"), "") if a["type"] == "issue_warning" else f"DISPATCH {a['crew_id']}"
        print(f"[{a['id']}] {what:<14} {a['zone_id']:<11} | {a['text']}")
        print(f"     evidence: {a['evidence']}")


def ask_approval():
    raw = input("\nApprove which actions? (all / none / e.g. 1,3): ").strip().lower()
    if raw in ("all", "none"):
        return raw
    return [int(x) for x in raw.replace(" ", "").split(",") if x.isdigit()]


async def main(question: str, auto_approve: bool = False) -> dict:
    tools = await mcp_client().get_tools()
    graph = build_graph(tools)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": langfuse_callbacks()}

    state = await graph.ainvoke({"question": question}, config)
    if "__interrupt__" in state:
        proposal = state["__interrupt__"][0].value
        show(proposal)
        decision = "all" if auto_approve else ask_approval()
        state = await graph.ainvoke(Command(resume=decision), config)

    for r in state.get("results", []):
        print(f"  action {r['id']}: {'OK' if r['ok'] else 'FAILED'} {'' if r['ok'] else r['result']}")
    print("\n" + state["report"])
    flush_langfuse()
    return state


if __name__ == "__main__":
    load_dotenv(override=True)
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?", default="Assess all zones. Which need alerts or crews right now?")
    p.add_argument("--auto-approve", action="store_true")
    args = p.parse_args()
    asyncio.run(main(args.question, args.auto_approve))
