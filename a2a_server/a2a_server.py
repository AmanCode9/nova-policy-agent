"""
a2a_server.py  —  A2A (Agent-to-Agent) Server

This file wraps our LangGraph pipeline and exposes it as an HTTP server
following the A2A (Agent-to-Agent) protocol format.

What is A2A?
  A2A is a protocol by Google that lets AI agents talk to each other
  over HTTP. Any A2A-compatible agent (like Google ADK) can send a 
  request to this server and get a response.

The server exposes two main endpoints:
  GET  /                  → Agent Card (describes this agent's capabilities)
  POST /a2a               → Main endpoint to send queries and get answers

HOW TO RUN:
    cd a2a_server
    python a2a_server.py

The server starts at http://localhost:8000
"""

import sys
import os

# Make sure Python can find the langgraph_pipeline folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
PIPELINE_DIR = os.path.join(PROJECT_DIR, "langgraph_pipeline")
sys.path.insert(0, PIPELINE_DIR)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import time
import uuid

# Import our LangGraph pipeline
from graph import build_graph, run_query
from state import AgentState

# ── Create the FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="Policy Agent A2A Server",
    description="An A2A-compatible server wrapping a LangGraph policy Q&A pipeline",
    version="1.0.0",
)

# ── Build the LangGraph graph once at startup (not on every request) ──────────
print("[a2a_server] Loading LangGraph pipeline...")
graph_app = build_graph()
print("[a2a_server] Pipeline ready.")


# ─────────────────────────────────────────────────────────────────────────────
# A2A MESSAGE SCHEMAS
# These Pydantic models define the shape of requests and responses.
# ─────────────────────────────────────────────────────────────────────────────

class A2AMessage(BaseModel):
    """
    A single message in the A2A protocol.
    role: "user" or "agent"
    content: the text of the message
    """
    role: str
    content: str


class A2ARequest(BaseModel):
    """
    The incoming request body for the /a2a endpoint.
    
    id      : A unique ID for this request (client generates this)
    message : The user's message
    """
    id: Optional[str] = None
    message: A2AMessage


class A2AResponse(BaseModel):
    """
    The response body we send back.
    
    id      : Echo back the request id
    status  : "success" or "error"
    message : The agent's reply
    metadata: Extra info (query type, processing time, etc.)
    """
    id: str
    status: str
    message: A2AMessage
    metadata: dict


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — AGENT CARD
# This is a required A2A endpoint. It describes what this agent can do.
# Google ADK reads this to understand how to talk to our agent.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def agent_card():
    """
    Returns the Agent Card — a description of this agent.
    This is the A2A discovery endpoint.
    """
    return {
        "name": "Nova Technologies Policy Agent",
        "description": (
            "An AI agent that answers questions about Nova Technologies' "
            "company policies including HR (leave, code of conduct, onboarding) "
            "and administrative policies (travel, IT assets). "
            "Also handles general conversation."
        ),
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "multimodal": False,
        },
        "skills": [
            {
                "id": "policy_qa",
                "name": "Policy Q&A",
                "description": "Answer questions about company HR and administrative policies",
                "examples": [
                    "How many casual leaves do I get per year?",
                    "What is the notice period for a senior engineer?",
                    "What is the travel allowance for interns?",
                    "What happens if I lose my company laptop?",
                ],
            },
            {
                "id": "general_chat",
                "name": "General Chat",
                "description": "Friendly responses to greetings and casual questions",
                "examples": [
                    "Hi, how are you?",
                    "Good morning!",
                ],
            },
        ],
        "endpoints": {
            "query": "/a2a",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — MAIN QUERY ENDPOINT
# This is where the ADK agent sends user queries.
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/a2a")
async def handle_query(request: A2ARequest):
    """
    Main A2A query endpoint.
    
    Receives a user message, runs it through the LangGraph pipeline,
    and returns the agent's answer in A2A response format.
    """
    request_id = request.id or str(uuid.uuid4())
    user_text  = request.message.content
    start_time = time.time()

    print(f"\n[a2a_server] Received request {request_id}: '{user_text}'")

    try:
        # Build initial state for the LangGraph pipeline
        initial_state: AgentState = {
            "user_query":        user_text,
            "extraction_needed": "",
            "extraction_done":   False,
            "query_type":        "",
            "final_answer":      "",
            "error":             None,
        }

        # Run the LangGraph pipeline
        result = graph_app.invoke(initial_state)

        answer     = result.get("final_answer", "I could not generate an answer.")
        query_type = result.get("query_type", "unknown")
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        print(f"[a2a_server] Answered in {elapsed_ms}ms. Type: {query_type}")

        return A2AResponse(
            id=request_id,
            status="success",
            message=A2AMessage(
                role="agent",
                content=answer,
            ),
            metadata={
                "query_type":       query_type,
                "processing_time_ms": elapsed_ms,
                "extraction_done":  result.get("extraction_done", False),
            },
        )

    except Exception as e:
        print(f"[a2a_server] ERROR: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "id":     request_id,
                "status": "error",
                "message": {
                    "role":    "agent",
                    "content": f"An error occurred: {str(e)}",
                },
                "metadata": {},
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — HEALTH CHECK
# Useful for testing that the server is alive and the pipeline is loaded.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "pipeline": "loaded",
        "server":   "Policy Agent A2A Server v1.0.0",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Start the server
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Nova Technologies — Policy Agent A2A Server")
    print("="*55)
    print("  Endpoints:")
    print("    GET  http://localhost:8000/        → Agent Card")
    print("    POST http://localhost:8000/a2a     → Query Agent")
    print("    GET  http://localhost:8000/health  → Health Check")
    print("    GET  http://localhost:8000/docs    → Swagger UI")
    print("="*55 + "\n")

    uvicorn.run(
        "a2a_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # Set to True during development for auto-reload
        log_level="info",
    )
