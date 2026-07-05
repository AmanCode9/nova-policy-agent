"""
eval_dataset.py — Evaluation Dataset for Nova Policy Agent (R2 Cloud Version)
This file defines 10 test cases covering all 5 policy documents.
Each test case stores:
  - question       : the employee's question
  - expected_answer: the correct answer (fill in after first run — see below)
  - source_doc     : the .md object key in R2 that contains the answer
  - pdf_source     : the .pdf object key in R2 (for traceability)
  - category       : policy area label

WHAT CHANGED FROM THE LOCAL VERSION:
  - load_retrieval_context() now fetches .md content directly from Cloudflare R2
    instead of reading from a local md_outputs/ folder (which no longer exists).
  - pdf_source now stores the R2 object key (e.g. "hr_leave_policy.pdf"),
    not a local os.path.join path.

ADDED IN THIS VERSION:
  - MCP_TEST_CASES: test cases for evaluating MCP tools directly
    (md_reader.py and cache_checker.py) in isolation, used by test_mcp_tools.py.

HOW TO FILL IN expected_answer:
  1. Run the pipeline once:    python ../langgraph_pipeline/graph.py
  2. Review each answer against the actual policy PDFs in R2.
  3. Replace "FILL_IN_AFTER_FIRST_RUN" with the correct answer.
  4. Then run:                 deepeval test run test_policy_agent.py
"""

import os
import json
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# ── Path resolution & Environment ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# ── Cloudflare R2 S3 client ───────────────────────────────────────────────────
s3_client = boto3.client(
    service_name='s3',
    endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    region_name='auto',
)

BUCKET_NAME = os.getenv('R2_BUCKET_NAME', 'nova-policy-bucket')


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# 10 questions covering all 5 policy documents (2 per document).
# pdf_source is now the R2 object key, not a local path.
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [

    # ── 1. HR Leave Policy ────────────────────────────────────────────────────
    {
        "question": "How many casual leaves am I entitled to per year?",
        "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
        "source_doc": "hr_leave_policy.md",
        "pdf_source": "hr_leave_policy.pdf",          # R2 object key
        "category": "HR Leave",
    },
    # {
    #     "question": "Can I carry forward my earned leave to the next year, and if so, what is the maximum limit?",
    #     "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
    #     "source_doc": "hr_leave_policy.md",
    #     "pdf_source": "hr_leave_policy.pdf",
    #     "category": "HR Leave",
    # },

    # # ── 2. Code of Conduct ────────────────────────────────────────────────────
    # {
    #     "question": "What action should I take if I witness workplace harassment?",
    #     "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
    #     "source_doc": "hr_code_of_conduct.md",
    #     "pdf_source": "hr_code_of_conduct.pdf",
    #     "category": "Code of Conduct",
    # },
    # {
    #     "question": "Can I use company equipment and resources for personal use?",
    #     "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
    #     "source_doc": "hr_code_of_conduct.md",
    #     "pdf_source": "hr_code_of_conduct.pdf",
    #     "category": "Code of Conduct",
    # },

    # # ── 3. Travel & Expense Policy ────────────────────────────────────────────
    # {
    #     "question": "What is the hotel allowance for a manager traveling on company business?",
    #     "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
    #     "source_doc": "admin_travel_expense_policy.md",
    #     "pdf_source": "admin_travel_expense_policy.pdf",
    #     "category": "Travel & Expense",
    # },
    # {
    #     "question": "Within how many days must I submit a travel reimbursement claim?",
    #     "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
    #     "source_doc": "admin_travel_expense_policy.md",
    #     "pdf_source": "admin_travel_expense_policy.pdf",
    #     "category": "Travel & Expense",
    # },

    # # ── 4. IT Asset Policy ────────────────────────────────────────────────────
    # {
    #     "question": "What should I do immediately if I lose my company-issued laptop?",
    #     "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
    #     "source_doc": "admin_it_asset_policy.md",
    #     "pdf_source": "admin_it_asset_policy.pdf",
    #     "category": "IT Asset",
    # },
    # {
    #     "question": "Am I allowed to install personal software on a company device?",
    #     "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
    #     "source_doc": "admin_it_asset_policy.md",
    #     "pdf_source": "admin_it_asset_policy.pdf",
    #     "category": "IT Asset",
    # },

    # # ── 5. Onboarding / Offboarding Policy ────────────────────────────────────
    # {
    #     "question": "What documents must a new employee submit during the onboarding process?",
    #     "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
    #     "source_doc": "hr_onboarding_offboarding_policy.md",
    #     "pdf_source": "hr_onboarding_offboarding_policy.pdf",
    #     "category": "Onboarding",
    # },
    # {
    #     "question": "What is the notice period for a mid-level employee who resigns?",
    #     "expected_answer": "FILL_IN_AFTER_FIRST_RUN",
    #     "source_doc": "hr_onboarding_offboarding_policy.md",
    #     "pdf_source": "hr_onboarding_offboarding_policy.pdf",
    #     "category": "Offboarding",
    # },
]


# ─────────────────────────────────────────────────────────────────────────────
# MCP TEST CASES
# Used by test_mcp_tools.py to evaluate each MCP tool directly in isolation.
#
# WHY SEPARATE FROM TEST_CASES:
#   TEST_CASES evaluate the full LangGraph pipeline (question → final answer).
#   MCP_TEST_CASES evaluate individual MCP tool outputs before they reach
#   the pipeline — catching failures at the tool layer, not the answer layer.
#
# STRUCTURE:
#   tool             : which MCP tool is being tested
#   input            : argument passed to the tool (None for no-arg tools)
#   expected_contains: list of strings that must appear in the tool output
#   expected_value   : exact return value expected (used for check_cache)
#   description      : human-readable label shown in test output
# ─────────────────────────────────────────────────────────────────────────────

MCP_TEST_CASES = [

    # ── md_reader.py: list_policy_files ───────────────────────────────────────
    {
        "tool": "list_policy_files",
        "input": None,
        "expected_contains": [
            "hr_leave_policy.md",
            "hr_code_of_conduct.md",
            "admin_it_asset_policy.md",
            "admin_travel_expense_policy.md",
            "hr_onboarding_offboarding_policy.md",
        ],
        "expected_value": None,
        "description": "list_policy_files must return all 5 .md files from R2",
    },

    # ── md_reader.py: read_policy_file ────────────────────────────────────────
    {
        "tool": "read_policy_file",
        "input": "hr_leave_policy.md",
        "expected_contains": ["## Document Overview", "## Key Policies and Rules"],
        "expected_value": None,
        "description": "read_policy_file must return structured markdown with correct headings",
    },
    {
        "tool": "read_policy_file",
        "input": "admin_it_asset_policy.md",
        "expected_contains": ["## Document Overview", "## Key Policies and Rules"],
        "expected_value": None,
        "description": "read_policy_file must return structured markdown for IT asset policy",
    },

    # ── md_reader.py: search_policies ─────────────────────────────────────────
    {
        "tool": "search_policies",
        "input": "casual leave",
        "expected_contains": ["hr_leave_policy.md"],
        "expected_value": None,
        "description": "search_policies('casual leave') must return results from hr_leave_policy.md",
    },
    {
        "tool": "search_policies",
        "input": "laptop",
        "expected_contains": ["admin_it_asset_policy.md"],
        "expected_value": None,
        "description": "search_policies('laptop') must return results from admin_it_asset_policy.md",
    },
    {
        "tool": "search_policies",
        "input": "hotel",
        "expected_contains": ["admin_travel_expense_policy.md"],
        "expected_value": None,
        "description": "search_policies('hotel') must return results from admin_travel_expense_policy.md",
    },
    {
        "tool": "search_policies",
        "input": "harassment",
        "expected_contains": ["hr_code_of_conduct.md"],
        "expected_value": None,
        "description": "search_policies('harassment') must return results from hr_code_of_conduct.md",
    },
    {
        "tool": "search_policies",
        "input": "onboarding",
        "expected_contains": ["hr_onboarding_offboarding_policy.md"],
        "expected_value": None,
        "description": "search_policies('onboarding') must return results from hr_onboarding_offboarding_policy.md",
    },

    # ── md_reader.py: get_policy_summary ──────────────────────────────────────
    {
        "tool": "get_policy_summary",
        "input": "hr_leave_policy.md",
        "expected_contains": ["Overview", "Scope", "Purpose"],
        "expected_value": None,
        "description": "get_policy_summary must return Overview or Scope section for hr_leave_policy.md",
    },
    {
        "tool": "get_policy_summary",
        "input": "hr_code_of_conduct.md",
        "expected_contains": ["Overview", "Scope", "Purpose"],
        "expected_value": None,
        "description": "get_policy_summary must return Overview or Scope section for hr_code_of_conduct.md",
    },

    # ── cache_checker.py: check_cache ─────────────────────────────────────────
    {
        "tool": "check_cache",
        "input": None,
        "expected_contains": [],
        "expected_value": "no",
        "description": "check_cache must return 'no' after all PDFs are processed",
    },

    # ── cache_checker.py: get_cache_status ────────────────────────────────────
    {
        "tool": "get_cache_status",
        "input": None,
        "expected_contains": [
            "Total PDFs in bucket",
            "Already processed",
            "Pending (not cached)",
        ],
        "expected_value": None,
        "description": "get_cache_status must return a complete status summary",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Fetch retrieval context directly from R2
#
# DeepEval's FaithfulnessMetric and AnswerRelevancyMetric require
# retrieval_context as a list of strings — one string per retrieved chunk.
# Since our pipeline loads each full .md file as one context block,
# we return a single-item list containing the full document content.
#
# WHAT CHANGED: Previously read from local md_outputs/ folder.
# Now streams the .md file directly from the Cloudflare R2 bucket.
# ─────────────────────────────────────────────────────────────────────────────

def load_retrieval_context(source_doc: str) -> list:
    """
    Fetches a .md policy file from R2 and returns its content as list[str].
    This represents the 'retrieved chunks' that policy_agent_node used to
    generate its answer — directly maps to Step 2 of Agent-Level Eval.

    Args:
        source_doc: R2 object key of the .md file, e.g. "hr_leave_policy.md"

    Returns:
        List with one string: the full content of that policy document.

    Raises:
        FileNotFoundError if the .md file hasn't been generated/uploaded yet.
    """
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=source_doc)
        content  = response["Body"].read().decode("utf-8")
        print(f"[eval] Loaded from R2: {source_doc} ({len(content)} characters)")
        return [content]   # DeepEval expects list[str]

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "NoSuchKey":
            raise FileNotFoundError(
                f"\n[eval] Policy document not found in R2: {source_doc}\n"
                f"[eval] Run the PDF extractor first so it uploads the .md to R2:\n"
                f"       cd langgraph_pipeline && python graph.py\n"
            )
        raise RuntimeError(f"[eval] R2 error fetching {source_doc}: {e}")