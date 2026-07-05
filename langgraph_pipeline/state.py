"""
state.py  —  Shared State for the LangGraph Pipeline

Every node in the graph reads from and writes to this state.
Think of it as the "memory" that travels through the entire pipeline.

LangGraph uses TypedDict so that each field is clearly typed.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    This is the state that flows through every node in our LangGraph pipeline.

    Fields:
        user_query      : The question the user asked.
        extraction_needed : "yes" or "no" from the cache checker.
        extraction_done  : True once the PDF extractor has run.
        query_type      : "policy" or "general" — decided by the orchestrator.
        final_answer    : The final answer to send back to the user.
        error           : Optional error message if something goes wrong.
    """

    user_query        : str
    extraction_needed : str          # "yes" or "no"
    extraction_done   : bool
    query_type        : str          # "policy" or "general"
    final_answer      : str
    error             : Optional[str]
