"""
test_policy_agent.py — DeepEval Evaluation Suite for Nova Policy Agent (R2 Cloud Version)

This file evaluates the full LangGraph pipeline using DeepEval.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT-LEVEL EVALUATION

  The policy_agent_node performs these steps internally:
    Step 1 — Read PDF        → pdf_extractor_node writes .md files to R2
    Step 2 — Retrieve chunks → policy_agent_node loads all .md files from R2
    Step 3 — Reason          → LLM processes context + question
    Step 4 — Generate answer → stored in state["final_answer"]

  DeepEval maps directly onto this:
    retrieval_context = content of the .md file fetched from R2  (Step 2 output)
    actual_output     = state["final_answer"]                     (Step 4 output)
    input             = user question
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT CHANGED FROM LOCAL VERSION:
  - load_retrieval_context() now fetches .md content from R2 (via eval_dataset.py).
  - run_pipeline_for_eval() bypasses cache_check_node and pdf_extractor_node
    to avoid unnecessary R2 list calls on every test — evaluation only needs
    the orchestrator → policy_agent path.

METRICS USED:
  Metric 1 — AnswerRelevancyMetric  : Is the answer relevant to the question?
  Metric 2 — FaithfulnessMetric     : Is the answer grounded in the .md document?
  Metric 3 — HallucinationMetric    : Does the answer contradict the context?
  Metric 4 — GEval (PolicyAccuracy) : LLM-as-judge policy correctness check

SETUP:
  pip install deepeval boto3
  # Configure .env with:
  #   GOOGLE_API_KEY        (used by both the pipeline LLM and the DeepEval judge LLM)
  #   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME

HOW TO RUN (pytest mode):
  cd evaluation
  deepeval test run test_policy_agent.py

HOW TO RUN (standalone):
  cd evaluation
  python test_policy_agent.py
"""

import os
import sys
import time
import pytest
from dotenv import load_dotenv
load_dotenv()

# ── Add langgraph_pipeline to path ────────────────────────────────────────────
# graph.py, state.py, and nodes.py live in a sibling folder (langgraph_pipeline/),
# not in evaluation/ — Python won't find them without this.
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(BASE_DIR)
PIPELINE_DIR = os.path.join(PROJECT_DIR, "langgraph_pipeline")
sys.path.insert(0, PIPELINE_DIR)

# ── DeepEval imports ──────────────────────────────────────────────────────────
from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams as LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.models import GeminiModel

# ── Pipeline imports ──────────────────────────────────────────────────────────
from graph import build_graph
from state import AgentState
from nodes import orchestrator_node, policy_agent_node, general_agent_node
from eval_dataset import TEST_CASES, load_retrieval_context


# ─────────────────────────────────────────────────────────────────────────────
# SETUP: Build the graph ONCE and reuse across all test cases
# ─────────────────────────────────────────────────────────────────────────────

print("\n[eval] Loading LangGraph pipeline...")
pipeline = build_graph()
print("[eval] Pipeline ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Run only the query-answering part of the pipeline for evaluation.
#
# Skips cache_check_node and pdf_extractor_node — evaluation only needs the
# orchestrator → agent path, since PDFs are assumed already extracted to R2.
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_for_eval(question: str) -> str:
    """
    Runs only the orchestrator and agent nodes — skips cache check and
    PDF extraction. This is the correct scope for agent-level evaluation.

    Returns the final answer string.
    """
    state: AgentState = {
        "user_query":        question,
        "extraction_needed": "no",
        "extraction_done":   True,  
        "query_type":        "",
        "final_answer":      "",
        "error":             None,
    }

    # Step 1: Classify the query
    state = orchestrator_node(state)

    # Step 2: Route to the right agent
    if state.get("query_type") == "policy":
        state = policy_agent_node(state)
    else:
        state = general_agent_node(state)

    return state.get("final_answer", "No answer generated.")


# ─────────────────────────────────────────────────────────────────────────────
# METRIC DEFINITIONS
# API key is read from GOOGLE_API_KEY in .env — never hardcode it here.
# ─────────────────────────────────────────────────────────────────────────────

model = GeminiModel(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# Metric 1 — Answer Relevancy
answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=model,
    include_reason=True,
)

# Metric 2 — Faithfulness
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=model,
    include_reason=True,
)

# Metric 3 — Hallucination
hallucination_metric = HallucinationMetric(
    threshold=0.5,
    model=model,
)

# Metric 4 — GEval (PolicyAnswerAccuracy)
policy_accuracy_metric = GEval(
    name="PolicyAnswerAccuracy",
    model=model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[
        "Check if the actual output directly and clearly answers the employee's policy question.",
        "Verify the actual output does not contradict any fact present in the expected output.",
        "Penalize answers that only say 'contact HR' without providing the actual policy details.",
        "Give full credit if the factual content is correct, even if the wording differs.",
        "Penalize answers that include information not supported by company policy.",
    ],
    threshold=0.6,
)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — HR Leave Policy
# Metrics: AnswerRelevancy
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "case",
    [c for c in TEST_CASES if c["category"] == "HR Leave"]
)
def test_hr_leave_policy(case):
    """Evaluates policy agent answers about HR Leave."""
    print(f"\n[TEST] HR Leave | Q: {case['question']}")
    print(f"       R2 PDF source: {case['pdf_source']}")

    actual_output     = run_pipeline_for_eval(case["question"])
    retrieval_context = load_retrieval_context(case["source_doc"])

    print(f"[TEST] Answer: {actual_output[:120]}...")

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        expected_output=case["expected_answer"],
        retrieval_context=retrieval_context,
    )

    assert_test(test_case, [answer_relevancy_metric])


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Code of Conduct
# Metrics: AnswerRelevancy
# Note: HallucinationMetric uses context= not retrieval_context=
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "case",
    [c for c in TEST_CASES if c["category"] == "Code of Conduct"]
)
def test_code_of_conduct(case):
    """Evaluates policy agent answers about Code of Conduct."""
    print(f"\n[TEST] Code of Conduct | Q: {case['question']}")
    print(f"       R2 PDF source: {case['pdf_source']}")

    actual_output     = run_pipeline_for_eval(case["question"])
    retrieval_context = load_retrieval_context(case["source_doc"])

    print(f"[TEST] Answer: {actual_output[:120]}...")

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        context=retrieval_context,           
        retrieval_context=retrieval_context,
    )

    assert_test(test_case, [answer_relevancy_metric])


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Admin Policies (Travel & IT Asset)
# Metrics: Faithfulness + GEval
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "case",
    [c for c in TEST_CASES if c["category"] in ("Travel & Expense", "IT Asset")]
)
def test_admin_policies(case):
    """Evaluates Travel & Expense and IT Asset policy answers."""
    print(f"\n[TEST] Admin Policy ({case['category']}) | Q: {case['question']}")
    print(f"       R2 PDF source: {case['pdf_source']}")

    actual_output     = run_pipeline_for_eval(case["question"])
    retrieval_context = load_retrieval_context(case["source_doc"])

    print(f"[TEST] Answer: {actual_output[:120]}...")

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        expected_output=case["expected_answer"],
        retrieval_context=retrieval_context,
    )

    # GEval requires expected_output — skip if not yet filled in
    metrics = [faithfulness_metric]
    if case["expected_answer"] != "FILL_IN_AFTER_FIRST_RUN":
        metrics.append(policy_accuracy_metric)

    assert_test(test_case, metrics)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Onboarding / Offboarding
# Metrics: All 4 — full agent-level evaluation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "case",
    [c for c in TEST_CASES if c["category"] in ("Onboarding", "Offboarding")]
)
def test_onboarding_offboarding(case):
    """
    Evaluates Onboarding and Offboarding policy answers.
    Runs all 4 DeepEval metrics — most complete agent-level evaluation.

    Agent-Level Evaluation map:
      Step 1 (Read PDF)        → pdf_source traces back to original R2 PDF key
      Step 2 (Retrieve chunks) → retrieval_context = .md content fetched from R2
      Step 3 (Reason)          → LLM internal reasoning (not directly measured)
      Step 4 (Generate answer) → actual_output = state["final_answer"]
    """
    print(f"\n[TEST] {case['category']} | Q: {case['question']}")
    print(f"       R2 PDF source: {case['pdf_source']}")

    actual_output     = run_pipeline_for_eval(case["question"])
    retrieval_context = load_retrieval_context(case["source_doc"])

    print(f"[TEST] Answer: {actual_output[:120]}...")

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        expected_output=case["expected_answer"],
        context=retrieval_context,            
        retrieval_context=retrieval_context,  
    )

    # GEval only runs once expected_answers are filled in
    metrics = [answer_relevancy_metric, faithfulness_metric, hallucination_metric]
    if case["expected_answer"] != "FILL_IN_AFTER_FIRST_RUN":
        metrics.append(policy_accuracy_metric)

    assert_test(test_case, metrics)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE RUNNER — no pytest needed
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Nova Technologies — Policy Agent DeepEval Evaluation")
    print("=" * 65)
    print("  Metrics: AnswerRelevancy, Faithfulness, Hallucination, GEval")
    print("=" * 65)

    all_test_cases = []
    skipped        = []

    for case in TEST_CASES:
        print(f"\n  [{case['category']}]")
        print(f"  Question     : {case['question']}")
        print(f"  R2 PDF source: {case['pdf_source']}")

        try:
            retrieval_context = load_retrieval_context(case["source_doc"])
            actual_output     = run_pipeline_for_eval(case["question"])
            time.sleep(15)
            print(f"  Answer       : {actual_output[:100]}...")

            test_case = LLMTestCase(
                input=case["question"],
                actual_output=actual_output,
                expected_output=case["expected_answer"],
                context=retrieval_context,
                retrieval_context=retrieval_context,
            )
            all_test_cases.append(test_case)

        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            skipped.append(case["question"])
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append(case["question"])

    if all_test_cases:
        print(f"\n{'=' * 65}")
        print(f"  Running DeepEval on {len(all_test_cases)} test cases...")
        if skipped:
            print(f"  Skipped: {len(skipped)} cases (.md files not found in R2)")
        print(f"{'=' * 65}\n")

        evaluate(
            test_cases=all_test_cases,
            metrics=[
                answer_relevancy_metric,
                faithfulness_metric,
                hallucination_metric,
            ],
        )
    else:
        print("\n  No test cases could be run.")
        print("  Run the PDF extractor first: python langgraph_pipeline/graph.py")