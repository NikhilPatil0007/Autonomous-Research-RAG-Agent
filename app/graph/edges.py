from __future__ import annotations

from app.config import get_settings
from app.graph.state import ResearchState
from app.observability import get_logger

log = get_logger("app.graph.edges")


def after_evidence_analyzer(state: ResearchState) -> str:
    """Rewrite+retrieve while under cap; otherwise synthesize."""
    max_iter = get_settings().max_research_iterations
    iteration = int(state.get("iteration") or 0)
    needs_more = bool(state.get("needs_more_research"))
    if needs_more and iteration < max_iter:
        route = "query_rewriter"
    else:
        route = "synthesizer"
    log.info(
        "edge.after_evidence_analyzer",
        extra={
            "component": "graph",
            "event": "edge.route",
            "from_node": "evidence_analyzer",
            "route": route,
            "needs_more_research": needs_more,
            "iteration": iteration,
            "max_iter": max_iter,
        },
    )
    return route


def after_critic(state: ResearchState) -> str:
    """One critic-driven retrieve if fail and not already used."""
    critic_pass = bool(state.get("critic_pass"))
    used = bool(state.get("critic_retrieve_used"))
    if critic_pass or used:
        route = "finalize"
    else:
        route = "researcher"
    log.info(
        "edge.after_critic",
        extra={
            "component": "graph",
            "event": "edge.route",
            "from_node": "critic",
            "route": route,
            "critic_pass": critic_pass,
            "critic_retrieve_used": used,
        },
    )
    return route
