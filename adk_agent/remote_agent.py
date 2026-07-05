"""
remote_agent.py  —  Google ADK Remote Agent

This file creates a Google ADK agent that acts as the "bridge" between
the A2UI frontend and our LangGraph A2A server.

HOW IT WORKS:
  1. The A2UI frontend (CopilotKit) sends a user message to this ADK agent.
  2. This ADK agent forwards the message to our A2A server at localhost:8000.
  3. The A2A server runs it through LangGraph and returns the answer.
  4. This ADK agent receives the answer and returns it to A2UI.

WHY ADK?
  Google ADK (Agent Development Kit) is a framework that makes it easy to:
  - Call remote A2A-compatible agents
  - Connect agents to A2UI frontends
  - Add tool use, memory, and multi-agent routing

REQUIREMENTS:
  pip install google-adk requests

HOW TO RUN:
  Option 1 — Run with ADK CLI (starts a web server for the agent):
    adk web

  Option 2 — Run directly to test:
    python remote_agent.py
"""

import os
import json
import requests
import uuid
from dotenv import load_dotenv

# Load environment variables (.env file in the project root)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# ── A2A Server configuration ─────────────────────────────────────────────────
A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Call the A2A server
# ─────────────────────────────────────────────────────────────────────────────

def call_a2a_server(user_message: str) -> dict:
    """
    Sends a user message to the LangGraph A2A server and returns the response.

    Args:
        user_message: The text the user typed.

    Returns:
        A dict with 'answer', 'query_type', and 'processing_time_ms'.
    """
    request_id = str(uuid.uuid4())

    payload = {
        "id": request_id,
        "message": {
            "role":    "user",
            "content": user_message,
        },
    }

    try:
        response = requests.post(
            f"{A2A_SERVER_URL}/a2a",
            json=payload,
            timeout=60,   # 60 second timeout (Gemini can be slow sometimes)
        )
        response.raise_for_status()
        data = response.json()

        return {
            "answer":            data["message"]["content"],
            "query_type":        data["metadata"].get("query_type", "unknown"),
            "processing_time_ms": data["metadata"].get("processing_time_ms", 0),
            "status":            data["status"],
        }

    except requests.exceptions.ConnectionError:
        return {
            "answer": (
                "Could not connect to the policy server. "
                "Please make sure the A2A server is running at " + A2A_SERVER_URL
            ),
            "query_type":        "error",
            "processing_time_ms": 0,
            "status":            "error",
        }

    except Exception as e:
        return {
            "answer":            f"An error occurred: {str(e)}",
            "query_type":        "error",
            "processing_time_ms": 0,
            "status":            "error",
        }


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE ADK AGENT DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

try:
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool

    # ── Define the tool that ADK will use to call our A2A server ──────────────

    def query_policy_agent(user_question: str) -> str:
        """
        Sends the user's question to the Nova Technologies Policy Agent
        (running as an A2A server) and returns the answer.

        Use this tool for ALL questions — whether they are about company
        policies or just casual conversation. The policy agent handles both.

        Args:
            user_question: The full text of what the user asked.

        Returns:
            The agent's answer as a string.
        """
        result = call_a2a_server(user_question)
        return result["answer"]


    # ── Create the ADK agent ──────────────────────────────────────────────────

    policy_agent = Agent(
        # Model to use for the ADK agent's own reasoning
        model="gemini-1.5-flash",

        # This name identifies the agent in ADK and A2UI
        name="nova_policy_agent",

        # Instruction tells the ADK agent how to behave
        instruction="""
You are the virtual assistant for Nova Technologies Private Limited.
Your name is Nova Assistant.

You help employees with:
1. Questions about company HR policies (leave, code of conduct, onboarding/offboarding)
2. Questions about administrative policies (travel expenses, IT assets)
3. Casual conversation and greetings

For every user message, use the query_policy_agent tool to get the answer.
Then present that answer to the user in a friendly, clear way.

Always be professional, warm, and helpful.
If the answer involves a specific number or policy rule, make sure to 
repeat it clearly so the user doesn't miss it.
""",

        # Give the agent access to our policy query tool
        tools=[FunctionTool(query_policy_agent)],
    )

    print("[remote_agent] Google ADK agent created successfully.")
    ADK_AVAILABLE = True

except ImportError:
    print("[remote_agent] Google ADK not installed.")
    print("  Install it with: pip install google-adk")
    print("  Falling back to direct A2A server calls for testing.\n")
    ADK_AVAILABLE = False
    policy_agent  = None


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST MODE
# Run this file directly to test the A2A server connection without ADK.
# ─────────────────────────────────────────────────────────────────────────────

def test_direct_connection():
    """
    Tests the connection to the A2A server directly (without ADK).
    Useful for verifying that the A2A server is running correctly.
    """
    print("\n" + "="*55)
    print("  Testing direct connection to A2A server...")
    print(f"  Server URL: {A2A_SERVER_URL}")
    print("="*55)

    # First check the health endpoint
    try:
        health = requests.get(f"{A2A_SERVER_URL}/health", timeout=5)
        print(f"  Health check: {health.json()['status']}")
    except Exception as e:
        print(f"  Health check FAILED: {e}")
        print("  Make sure to run: python a2a_server/a2a_server.py first")
        return

    # Test queries
    test_queries = [
        "Hello!",
        "How many earned leaves can I carry forward?",
        "What is the hotel allowance for a manager on a business trip?",
        "What should I do if I lose my company laptop?",
    ]

    for query in test_queries:
        print(f"\n  Query   : {query}")
        result = call_a2a_server(query)
        print(f"  Type    : {result['query_type']}")
        print(f"  Time    : {result['processing_time_ms']}ms")
        print(f"  Answer  : {result['answer'][:150]}...")


if __name__ == "__main__":
    test_direct_connection()
