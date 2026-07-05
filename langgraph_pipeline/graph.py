"""
graph.py  —  The LangGraph Pipeline (Main File)

This file builds and compiles the complete LangGraph graph.
It connects all the nodes with edges and conditional routing.

HOW THE GRAPH FLOWS:
  START
    │
    ▼
  cache_check_node          ← checks if PDFs need extraction
    │
    ├─ "yes" ──► pdf_extractor_node   ← extracts PDFs to .md
    │                │
    │                ▼
    └─ "no"  ──► orchestrator_node    ← classifies user query
                     │
                     ├─ "policy"  ──► policy_agent_node   ← answers from .md docs
                     │
                     └─ "general" ──► general_agent_node  ← answers directly
                                            │
                                           END

RUN THIS FILE to test the pipeline:
    python graph.py
"""

import logging
import time

from langgraph.graph import StateGraph, END

from state import AgentState
from nodes import (
    cache_check_node,
    pdf_extractor_node,
    orchestrator_node,
    policy_agent_node,
    general_agent_node,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
# These are the "condition" functions for conditional edges.
# They read from the state and return a string that LangGraph
# uses to decide which node to go to next.
# ─────────────────────────────────────────────────────────────────────────────

def route_after_cache_check(state: AgentState) -> str:
    """
    After the cache check node, decide:
      - "yes" → go to pdf_extractor_node  (new PDFs found)
      - "no"  → go to orchestrator_node   (all PDFs already processed)
    """
    return state.get("extraction_needed", "no")


def route_after_orchestrator(state: AgentState) -> str:
    """
    After the orchestrator node, decide:
      - "policy"  → go to policy_agent_node
      - "general" → go to general_agent_node
    """
    return state.get("query_type", "general")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Creates and compiles the LangGraph StateGraph.

    Returns:
        A compiled LangGraph app, ready to be invoked with .invoke(state).

    Raises:
        Exception: re-raised if graph compilation fails, after logging
                   the error — a broken graph should never be silently
                   swallowed since nothing downstream can function without it.
    """
    try:
        # Step 1: Create a new graph using our AgentState as the schema
        graph = StateGraph(AgentState)

        # Step 2: Add all nodes (each node is a function from nodes.py)
        graph.add_node("cache_check", cache_check_node)
        graph.add_node("pdf_extractor", pdf_extractor_node)
        graph.add_node("orchestrator", orchestrator_node)
        graph.add_node("policy_agent", policy_agent_node)
        graph.add_node("general_agent", general_agent_node)

        # Step 3: Set the entry point — graph always starts at cache_check
        graph.set_entry_point("cache_check")

        # Step 4: Add CONDITIONAL edge after cache_check
        # Depending on what route_after_cache_check() returns,
        # the graph goes to a different node.
        graph.add_conditional_edges(
            "cache_check",              # From this node
            route_after_cache_check,    # Call this function to decide
            {
                "yes": "pdf_extractor",  # New PDFs found → extract them
                "no": "orchestrator",    # Already processed → classify query
            },
        )

        # Step 5: After pdf_extractor runs, always go to orchestrator
        graph.add_edge("pdf_extractor", "orchestrator")

        # Step 6: Add CONDITIONAL edge after orchestrator
        # Depending on what route_after_orchestrator() returns,
        # the graph goes to either the policy agent or the general agent.
        graph.add_conditional_edges(
            "orchestrator",               # From this node
            route_after_orchestrator,     # Call this function to decide
            {
                "policy": "policy_agent",    # Policy question → policy agent
                "general": "general_agent",  # General question → general agent
            },
        )

        # Step 7: Both agents go to END after giving their answer
        graph.add_edge("policy_agent", END)
        graph.add_edge("general_agent", END)

        # Step 8: Compile the graph (this validates structure and locks it in)
        compiled_graph = graph.compile()
        logger.info("Graph compiled successfully.")
        return compiled_graph

    except Exception:
        logger.exception("Failed to build/compile the LangGraph pipeline.")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# RUN A SINGLE QUERY THROUGH THE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_query(user_query: str, app=None) -> str:
    """
    Runs a single question through the full pipeline and returns the answer.

    Args:
        user_query: The question to ask.
        app: Optional pre-compiled graph (from build_graph()) to reuse across
             multiple calls instead of rebuilding it every time.

    Returns:
        The final answer string. If the pipeline itself raises, a safe
        fallback message is returned instead of propagating the exception —
        callers (API endpoints, batch scripts) shouldn't crash on a single
        bad query.
    """
    if app is None:
        app = build_graph()

    initial_state: AgentState = {
        "user_query": user_query,
        "extraction_needed": "",
        "extraction_done": False,
        "query_type": "",
        "final_answer": "",
        "error": None,
    }

    logger.info("Running query: %s", user_query)

    try:
        result = app.invoke(initial_state)
    except Exception:
        logger.exception("Pipeline execution failed for query: %s", user_query)
        return (
            "Something went wrong while processing your question. "
            "Please try again in a moment."
        )

    final_answer = result.get("final_answer", "No answer generated.")
    logger.info("Answer generated (%d characters).", len(final_answer))
    return final_answer


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — test queries when running this file directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Building and testing the policy agent pipeline...")

    # Build the graph once and reuse it for all test queries
    pipeline_app = build_graph()

    test_queries = [
        "Hi, good morning!",
        "How many casual leaves do I get in a year?",
        "What is the notice period for a senior engineer?",
        "What is the travel allowance for interns?",
        "Can I carry forward my earned leave?",
        "What happens if I lose my company laptop?",
    ]

    # Pause between queries to stay within the Gemini API's rate limits
    QUERY_DELAY_SECONDS = 15

    for i, query in enumerate(test_queries):
        answer = run_query(query, pipeline_app)
        print(f"\n{'=' * 60}")
        print(f"  Q: {query}")
        print(f"{'-' * 60}")
        print(f"  A: {answer}")
        print(f"{'=' * 60}\n")

        if i < len(test_queries) - 1:
            time.sleep(QUERY_DELAY_SECONDS)