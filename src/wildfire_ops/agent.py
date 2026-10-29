"""LangGraph agent: investigate -> human_review (interrupt) -> execute -> report.

The agent talks to the MCP server like any other MCP client would.
It can only READ during investigation; ACTION tools run after a human approves.
"""
import os
import sys
from typing import Literal, Optional, TypedDict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from wildfire_ops.data_sources import ROOT

READ_TOOLS = {"list_zones", "get_weather_risk", "get_active_fires", "assess_zone_risk"}
ACTION_TOOLS = {"issue_warning", "dispatch_crew"}

SYSTEM_PROMPT = """You support a county emergency manager in the East Bay Hills during fire season.

Process:
1. Call list_zones, then assess_zone_risk for EVERY zone. Use other read tools only if you need detail.
2. Propose actions. Use the recommended_level from assess_zone_risk as your default.
   Deviate only with a concrete reason from the data (e.g. population, egress notes), and say so in evidence.
3. Propose issue_warning for every zone at advisory or above. Never warn a zone at 'monitor'.
4. Propose dispatch_crew for the highest-risk zone(s) with an active fire, preferring crews based nearby.
5. Never invent data. If a tool fails, say what is missing in the situation summary.

You cannot execute actions. A human will approve them."""


class ProposedAction(BaseModel):
    id: int = Field(description="1, 2, 3...")
    type: Literal["issue_warning", "dispatch_crew"]
    zone_id: str
    level: Optional[Literal["advisory", "evacuation_warning", "evacuation_order"]] = Field(
        default=None, description="Required for issue_warning")
    crew_id: Optional[str] = Field(default=None, description="Required for dispatch_crew")
    text: str = Field(description="Public alert message, or reason for the dispatch")
    evidence: str = Field(description="Scores and data points that justify this action")


class Proposal(BaseModel):
    situation_summary: str = Field(description="3-5 sentences for the emergency manager")
    actions: list[ProposedAction]


class State(TypedDict, total=False):
    question: str
    proposal: dict
    approved_ids: list[int]
    results: list[dict]
    report: str


def mcp_client() -> MultiServerMCPClient:
    """Spawns our MCP server over stdio. Env is passed through (API keys, WILDFIRE_SCENARIO)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return MultiServerMCPClient({
        "wildfire": {"command": sys.executable, "args": ["-m", "wildfire_ops.mcp_server"],
                     "transport": "stdio", "env": env}
    })


def build_graph(tools: list, model=None, investigate_fn=None):
    tools_by_name = {t.name: t for t in tools}

    async def investigate(state: State) -> State:
        if investigate_fn:  # used in tests
            return {"proposal": await investigate_fn(state)}
        llm = model or init_chat_model(os.getenv("MODEL", "anthropic:claude-sonnet-5-5"))
        investigator = create_agent(
            llm, tools=[t for t in tools if t.name in READ_TOOLS],
            system_prompt=SYSTEM_PROMPT, response_format=Proposal, name="investigator")
        out = await investigator.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
        return {"proposal": out["structured_response"].model_dump()}

    def human_review(state: State) -> State:
        # Pauses the graph. Resume with Command(resume=[ids]) / "all" / "none".
        decision = interrupt(state["proposal"])
        all_ids = [a["id"] for a in state["proposal"]["actions"]]
        if decision == "all":
            ids = all_ids
        elif decision == "none":
            ids = []
        else:
            ids = [i for i in decision if i in all_ids]
        return {"approved_ids": ids}

    async def execute(state: State) -> State:
        results = []
        for a in state["proposal"]["actions"]:
            if a["id"] not in state.get("approved_ids", []):
                continue
            if a["type"] == "issue_warning":
                args = {"zone_id": a["zone_id"], "level": a["level"], "message": a["text"]}
            else:
                args = {"crew_id": a["crew_id"], "zone_id": a["zone_id"], "reason": a["text"]}
            try:
                out = await tools_by_name[a["type"]].ainvoke(args)
                results.append({"id": a["id"], "ok": True, "result": str(out)[:300]})
            except Exception as e:  # keep going, report the failure
                results.append({"id": a["id"], "ok": False, "result": str(e)})
        return {"results": results}

    def report(state: State) -> State:
        n_prop = len(state["proposal"]["actions"])
        n_ok = sum(r["ok"] for r in state.get("results", []))
        n_appr = len(state.get("approved_ids", []))
        return {"report": f"{n_prop} actions proposed, {n_appr} approved, {n_ok} executed successfully."}

    g = StateGraph(State)
    g.add_node("investigate", investigate)
    g.add_node("human_review", human_review)
    g.add_node("execute", execute)
    g.add_node("report", report)
    g.add_edge(START, "investigate")
    g.add_edge("investigate", "human_review")
    g.add_edge("human_review", "execute")
    g.add_edge("execute", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=InMemorySaver())


def langfuse_callbacks() -> list:
    """Langfuse tracing if keys are set (LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL)."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return []
    from langfuse.langchain import CallbackHandler
    return [CallbackHandler()]


def flush_langfuse():
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse import get_client
        get_client().flush()
