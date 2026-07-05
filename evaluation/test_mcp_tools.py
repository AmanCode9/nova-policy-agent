"""
test_mcp_tools.py — MCP Tool Evaluation Suite for Nova Policy Agent

Evaluates the MCP server tools directly (not the full LangGraph pipeline).
Tests that each MCP tool returns correct, complete, and well-structured output.

TOOLS TESTED:
  From md_reader.py    : list_policy_files, read_policy_file,
                         search_policies, get_policy_summary
  From cache_checker.py: check_cache, get_cache_status

WHY TEST MCP TOOLS SEPARATELY:
  The LangGraph pipeline evaluation (test_policy_agent.py) tests the END result.
  But if an MCP tool returns wrong data, the pipeline answer will also be wrong.
  Testing MCP tools in isolation tells you EXACTLY which layer failed.

HOW TO RUN:
  cd evaluation
  pytest test_mcp_tools.py -v

  OR standalone:
  python test_mcp_tools.py
"""

import os
import sys
import pytest
from dotenv import load_dotenv
load_dotenv()

# ── Add mcp_tools to path ─────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(BASE_DIR)
MCP_DIR      = os.path.join(PROJECT_DIR, "mcp_tools")
sys.path.insert(0, MCP_DIR)

# ── Import MCP tool functions directly ───────────────────────────────────────
# We import the underlying functions, not the MCP server itself.
# This lets us call them like regular Python functions in tests.
from md_reader import (
    get_all_md_files,
    get_file_content,
    list_policy_files,
    read_policy_file,
    search_policies,
    get_policy_summary,
)
from cache_checker import (
    check_cache,
    get_cache_status,
    list_bucket_pdfs,
)

from eval_dataset import MCP_TEST_CASES


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 1 — list_policy_files
# Verifies that all 5 expected .md files are listed from R2
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_MD_FILES = [
    "hr_leave_policy.md",
    "hr_code_of_conduct.md",
    "admin_it_asset_policy.md",
    "admin_travel_expense_policy.md",
    "hr_onboarding_offboarding_policy.md",
]

def test_list_policy_files_returns_all_docs():
    """All 5 policy .md files must be present in R2."""
    result = list_policy_files()
    print(f"\n[MCP TEST] list_policy_files result:\n{result}")

    for expected_file in EXPECTED_MD_FILES:
        assert expected_file in result, (
            f"Expected '{expected_file}' in list_policy_files output, but got:\n{result}"
        )

def test_list_policy_files_not_empty():
    """list_policy_files must not return the 'no files found' message."""
    result = list_policy_files()
    assert "Run the PDF extractor first" not in result, (
        "list_policy_files returned empty — .md files not found in R2 bucket"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 2 — read_policy_file
# Verifies content is readable and has expected markdown structure
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename", EXPECTED_MD_FILES)
def test_read_policy_file_has_content(filename):
    """Each .md file must be readable and non-empty."""
    result = read_policy_file(filename)
    print(f"\n[MCP TEST] read_policy_file({filename}): {len(result)} chars")

    assert result is not None, f"read_policy_file({filename}) returned None"
    assert len(result) > 100, (
        f"read_policy_file({filename}) returned too little content: '{result[:50]}'"
    )
    assert "not found in bucket" not in result, (
        f"read_policy_file({filename}) says file not found in R2"
    )

@pytest.mark.parametrize("filename", EXPECTED_MD_FILES)
def test_read_policy_file_has_markdown_structure(filename):
    """Each .md file must contain the expected structured sections."""
    result = read_policy_file(filename)

    assert "## Document Overview" in result, (
        f"{filename} is missing '## Document Overview' section"
    )
    assert "## Key Policies and Rules" in result, (
        f"{filename} is missing '## Key Policies and Rules' section"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 3 — search_policies
# Verifies keyword search finds results in the right documents
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_CASES = [
    ("casual leave",  "hr_leave_policy.md"),
    ("laptop",        "admin_it_asset_policy.md"),
    ("hotel",         "admin_travel_expense_policy.md"),
    ("harassment",    "hr_code_of_conduct.md"),
    ("onboarding",    "hr_onboarding_offboarding_policy.md"),
]

@pytest.mark.parametrize("keyword,expected_source", SEARCH_CASES)
def test_search_policies_finds_keyword(keyword, expected_source):
    """search_policies must find the keyword and return the correct source doc."""
    result = search_policies(keyword)
    print(f"\n[MCP TEST] search_policies('{keyword}'):\n{result[:200]}")

    assert "No results found" not in result, (
        f"search_policies('{keyword}') found nothing — check if .md files are in R2"
    )
    assert expected_source in result, (
        f"search_policies('{keyword}') didn't mention '{expected_source}'\nGot:\n{result[:300]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 4 — get_policy_summary
# Verifies that summary returns Overview/Scope sections correctly
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename", EXPECTED_MD_FILES)
def test_get_policy_summary_has_overview(filename):
    """get_policy_summary must return Overview or Scope content."""
    result = get_policy_summary(filename)
    print(f"\n[MCP TEST] get_policy_summary({filename}):\n{result[:200]}")

    assert result is not None
    assert len(result) > 50, (
        f"get_policy_summary({filename}) returned too little content"
    )

    has_overview = any(
        kw in result for kw in ["Overview", "Scope", "Purpose"]
    )
    assert has_overview, (
        f"get_policy_summary({filename}) didn't return Overview/Scope/Purpose section.\n"
        f"Got: {result[:200]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 5 — check_cache (cache_checker.py)
# Verifies cache correctly reports all PDFs as processed
# ─────────────────────────────────────────────────────────────────────────────

def test_check_cache_returns_no_after_extraction():
    """
    After all PDFs are processed, check_cache must return 'no'.
    If it returns 'yes', the cache.json in R2 is out of sync.
    """
    result = check_cache()
    print(f"\n[MCP TEST] check_cache() returned: '{result}'")

    assert result == "no", (
        f"check_cache() returned '{result}' — expected 'no'.\n"
        f"This means some PDFs in R2 are not marked as processed in cache.json.\n"
        f"Run: python langgraph_pipeline/graph.py to process them first."
    )

def test_get_cache_status_shows_all_processed():
    """get_cache_status must show all 5 PDFs as processed."""
    result = get_cache_status()
    print(f"\n[MCP TEST] get_cache_status():\n{result}")

    assert "Pending (not cached) : 0" in result or "TODO" not in result, (
        f"get_cache_status shows pending PDFs — not all are processed.\nStatus:\n{result}"
    )
    assert "Total PDFs in bucket" in result


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Nova Technologies — MCP Tool Direct Evaluation")
    print("=" * 65)

    tests = [
        ("list_policy_files — all docs present",   test_list_policy_files_returns_all_docs),
        ("list_policy_files — not empty",           test_list_policy_files_not_empty),
        ("check_cache — returns no",                test_check_cache_returns_no_after_extraction),
        ("get_cache_status — all processed",        test_get_cache_status_shows_all_processed),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ PASS — {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL — {name}\n         {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR — {name}\n          {e}")
            failed += 1

    # Parametrized tests
    for filename in EXPECTED_MD_FILES:
        for test_fn, label in [
            (test_read_policy_file_has_content,        "read_policy_file — has content"),
            (test_read_policy_file_has_markdown_structure, "read_policy_file — has structure"),
            (test_get_policy_summary_has_overview,     "get_policy_summary — has overview"),
        ]:
            try:
                test_fn(filename)
                print(f"  ✓ PASS — {label} ({filename})")
                passed += 1
            except AssertionError as e:
                print(f"  ✗ FAIL — {label} ({filename})\n         {e}")
                failed += 1

    for keyword, expected_source in SEARCH_CASES:
        try:
            test_search_policies_finds_keyword(keyword, expected_source)
            print(f"  ✓ PASS — search_policies('{keyword}') → {expected_source}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL — search_policies('{keyword}')\n         {e}")
            failed += 1

    print(f"\n{'=' * 65}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    print(f"{'=' * 65}")