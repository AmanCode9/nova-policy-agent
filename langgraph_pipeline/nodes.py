"""
nodes.py  —  All Node Functions for the LangGraph Pipeline (R2 Cloud Version)

Each function here represents one "node" (one step) in the graph.
Nodes receive the current state, do their job, and return updated state fields.

NODES IN THIS FILE:
  1. cache_check_node      — Lists PDFs in R2; checks cloud cache for new ones
  2. pdf_extractor_node    — Downloads PDFs from R2, runs LLM, uploads .md back to R2
  3. orchestrator_node     — Classifies the user's query: "policy" or "general"
  4. policy_agent_node     — Streams .md files from R2, answers policy questions
  5. general_agent_node    — Answers general / casual questions directly

KEY IMPROVEMENTS OVER FIRST R2 DRAFT:
  - All R2 list calls use a paginator — safe for buckets with 1,000+ objects.
  - Temp-file path is initialised before the try/finally block so the finally
    clause never raises a NameError on a failed download.
  - Thinking-tag blocks (<think>…</think>) are stripped before the
    orchestrator evaluates the classification response.
  - policy_agent_node also uses the paginator for listing .md files.
"""

import os
import re
import json
import time
import boto3
from botocore.exceptions import ClientError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from state import AgentState

# ── Path configuration & Environment ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# ── Cloudflare R2 S3 client ──────────────────────────────────────────────────
s3_client = boto3.client(
    service_name='s3',
    endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    region_name='auto',
)

BUCKET_NAME = os.getenv('R2_BUCKET_NAME', 'nova-policy-bucket')
CACHE_KEY   = "cache.json"

# ── LLM setup ────────────────────────────────────────────────────────────────
# API key is read from the GOOGLE_API_KEY environment variable (set in .env).
# Never hardcode the key here — see .env.example for the required variables.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Strip <think>…</think> blocks some models emit in reasoning mode.
# Without this, the orchestrator's "policy"/"general" check can fail because
# the raw response starts with a multi-line thinking block before the answer.
# ─────────────────────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and return clean text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Paginated list of all object keys in the R2 bucket
# Using a paginator is mandatory — list_objects_v2 only returns up to 1,000
# keys per call, so a direct call silently drops anything beyond that limit.
# ─────────────────────────────────────────────────────────────────────────────

def _list_all_keys(suffix: str = "") -> list:
    """
    Returns every object key in the bucket that ends with `suffix`.
    Pass suffix=".pdf" or suffix=".md" to filter by type.
    """
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=BUCKET_NAME):
            for obj in page.get("Contents", []):
                if obj["Key"].lower().endswith(suffix.lower()):
                    keys.append(obj["Key"])
    except ClientError as e:
        print(f"[nodes] R2 list error: {e}")
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# R2 CLOUD CACHE HELPERS
# cache.json lives inside the same R2 bucket — no local file required.
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache_from_r2() -> dict:
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=CACHE_KEY)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return {"processed_files": []}
        print(f"[nodes] R2 cache get error: {e}")
        return {"processed_files": []}


def _save_cache_to_r2(cache: dict):
    try:
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=CACHE_KEY,
            Body=json.dumps(cache, indent=2),
            ContentType="application/json",
        )
    except ClientError as e:
        print(f"[nodes] R2 cache put error: {e}")


def _mark_processed_in_r2(filename: str):
    cache = _load_cache_from_r2()
    if filename not in cache["processed_files"]:
        cache["processed_files"].append(filename)
        _save_cache_to_r2(cache)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — CACHE CHECK NODE
# ─────────────────────────────────────────────────────────────────────────────

def cache_check_node(state: AgentState) -> AgentState:
    """
    Lists all PDFs in the R2 bucket (paginated) and compares them against the
    cloud cache to decide whether extraction is needed.

    Returns "yes" if new PDFs are found, "no" if everything is already done.
    """
    print("\n[NODE 1] Cache Check — checking R2 bucket for new PDFs...")

    try:
        all_pdfs = _list_all_keys(suffix=".pdf")
    except Exception as e:
        print(f"  R2 connection error: {e}")
        return {**state, "extraction_needed": "no", "error": str(e)}

    if not all_pdfs:
        print("  No PDFs found in the R2 bucket. Skipping extraction.")
        return {**state, "extraction_needed": "no"}

    cache = _load_cache_from_r2()
    processed = set(cache.get("processed_files", []))
    new_pdfs = [f for f in all_pdfs if f not in processed]

    if new_pdfs:
        print(f"  New PDFs detected: {new_pdfs}")
        print("  Decision: YES — run the PDF extractor")
        return {**state, "extraction_needed": "yes"}
    else:
        print(f"  All {len(all_pdfs)} PDFs already processed.")
        print("  Decision: NO — skip extraction, go to orchestrator")
        return {**state, "extraction_needed": "no"}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — PDF EXTRACTOR NODE
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """
You are a professional document analyst. You have been given the raw text 
extracted from a company policy PDF document.

Your task is to convert this raw text into a clean, well-structured Markdown file.
The output should be easy for an AI agent to search and answer questions from.

Structure the output EXACTLY in these sections (use ## for section headings):

## Document Overview
(1-2 sentences describing what this policy covers)

## Scope and Applicability
(Who does this policy apply to? Any exceptions?)

## Key Policies and Rules
(The main rules, listed clearly. Use bullet points.)

## Important Deadlines, Numbers, and Limits
(Any specific numbers like days, percentages, amounts, dates — listed clearly)

## Procedures and Processes
(Step-by-step processes mentioned in the document)

## Violations and Consequences
(What happens if someone breaks this policy?)

## Escalation and Contact Information
(Who to contact, email addresses, portal links)

---
RAW PDF TEXT:
{pdf_text}
---

Now write the structured Markdown output.
Do not include any introductory text before ## Document Overview.
"""


def pdf_extractor_node(state: AgentState) -> AgentState:
    """
    For each unprocessed PDF in R2:
      1. Downloads it to a local temp file.
      2. Extracts raw text with pypdf.
      3. Sends the text to the LLM for structured Markdown conversion.
      4. Uploads the .md file back to R2.
      5. Updates the cloud cache.

    The temp file is always cleaned up — even if an error occurs mid-download —
    because local_temp_pdf is assigned before the try/finally block.
    """
    print("\n[NODE 2] PDF Extractor — pulling PDFs from R2 and uploading .md outputs...")

    try:
        import pypdf
    except ImportError:
        print("  pypdf not installed. Run: pip install pypdf")
        return {**state, "extraction_done": False, "error": "pypdf not installed"}

    all_pdfs = _list_all_keys(suffix=".pdf")
    cache = _load_cache_from_r2()
    processed = set(cache.get("processed_files", []))
    new_pdfs = [f for f in all_pdfs if f not in processed]

    if not new_pdfs:
        print("  No new PDFs to process.")
        return {**state, "extraction_done": True}

    for pdf_filename in new_pdfs:
        print(f"\n  Processing: {pdf_filename}")

        # Assign path BEFORE try/finally so the finally clause is always safe
        local_temp_pdf = os.path.join(BASE_DIR, f"_temp_{pdf_filename}")
        pdf_text = ""

        try:
            # Step 1: Download PDF from R2
            s3_client.download_file(BUCKET_NAME, pdf_filename, local_temp_pdf)

            # Step 2: Extract raw text
            reader = pypdf.PdfReader(local_temp_pdf)
            pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            print(f"  Extracted {len(pdf_text)} characters.")

        except Exception as e:
            print(f"  ERROR downloading/reading {pdf_filename}: {e}")
            continue

        finally:
            # Always remove the temp file, whether or not extraction succeeded
            if os.path.exists(local_temp_pdf):
                os.remove(local_temp_pdf)

        if not pdf_text.strip():
            print(f"  WARNING: No text extracted from {pdf_filename}. Skipping.")
            continue

        # Step 3: Send to LLM for structured markdown
        try:
            prompt = EXTRACTION_PROMPT.format(pdf_text=pdf_text)
            response = llm.invoke([HumanMessage(content=prompt)])
            md_content = _strip_thinking(response.content)
            print(f"  LLM generated {len(md_content)} characters of markdown.")
        except Exception as e:
            print(f"  ERROR calling LLM for {pdf_filename}: {e}")
            continue

        # Step 4: Upload .md back to R2
        md_filename = pdf_filename.replace(".pdf", ".md")
        full_md_payload = f"<!-- Source: {pdf_filename} -->\n\n{md_content}"

        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=md_filename,
                Body=full_md_payload.encode("utf-8"),
                ContentType="text/markdown",
            )
            print(f"  Uploaded to R2: {md_filename}")
        except ClientError as e:
            print(f"  ERROR uploading {md_filename} to R2: {e}")
            continue

        # Step 5: Mark as processed in cloud cache
        _mark_processed_in_r2(pdf_filename)
        print(f"  Cache updated for: {pdf_filename}")

        print("  Sleeping 2 s to protect local model resources...")
        time.sleep(2)

    print("\n  All new PDFs processed successfully.")
    return {**state, "extraction_done": True}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — ORCHESTRATOR NODE
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFICATION_PROMPT_TEMPLATE = """
You are an intelligent routing assistant for a company policy Q&A system.

Read the user's query carefully and classify it into one of two categories:

1. "policy"  — The query is asking about company rules, HR policies, administrative
               procedures, leave, travel, IT assets, code of conduct, onboarding,
               offboarding, or anything that a company policy document would cover.

2. "general" — The query is casual, a greeting, small talk, a general knowledge
               question, or anything NOT related to company policies.

Examples of "policy" queries:
  - "How many casual leaves do I get per year?"
  - "What is the notice period for a senior engineer?"
  - "Can I carry forward my earned leave?"
  - "What happens if I lose my company laptop?"
  - "What is the travel allowance for managers?"

Examples of "general" queries:
  - "Hi, how are you?"
  - "What is the capital of France?"
  - "Tell me a joke"
  - "What is machine learning?"
  - "Good morning!"

User query: "{user_query}"

Reply with ONLY one word: either "policy" or "general". Nothing else.
"""


def orchestrator_node(state: AgentState) -> AgentState:
    """
    Classifies the user query as "policy" or "general".

    Some models may prepend a <think>…</think> block — _strip_thinking()
    removes it before we evaluate the one-word classification response.
    """
    print("\n[NODE 3] Orchestrator — classifying the user query...")
    user_query = state.get("user_query", "")
    print(f"  User query: {user_query}")

    classification_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(user_query=user_query)

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    raw_text = _strip_thinking(response.content)
    query_type = raw_text.strip().lower()

    if query_type not in ("policy", "general"):
        print(f"  Unexpected classification: '{query_type}'. Defaulting to 'general'.")
        query_type = "general"

    print(f"  Classification result: {query_type}")
    return {**state, "query_type": query_type}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — POLICY AGENT NODE
# ─────────────────────────────────────────────────────────────────────────────

POLICY_PROMPT_TEMPLATE = """
You are a helpful HR and administrative assistant for Nova Technologies Private Limited.
You have been given the company's official policy documents in Markdown format.

Your job is to answer the employee's question ONLY using the information 
found in these documents.

IMPORTANT RULES:
- Answer clearly and in simple language.
- Always mention which policy document your answer comes from.
- If the answer is not found in any of the documents, say:
  "I could not find information about this in the available policy documents.
   Please contact HR at hr@novatech.in for clarification."
- Do not make up any information.
- Keep the answer concise — under 200 words unless a detailed explanation is needed.

POLICY DOCUMENTS:
{combined_context}

EMPLOYEE QUESTION: {user_query}

Answer:
"""


def policy_agent_node(state: AgentState) -> AgentState:
    """
    Streams all .md policy files from R2 (paginated), builds a combined context
    block, and sends it to the LLM to answer the employee's question.
    """
    print("\n[NODE 4] Policy Agent — pulling markdown docs from R2...")
    user_query = state.get("user_query", "")

    # Paginated list of .md files
    md_files = _list_all_keys(suffix=".md")

    if not md_files:
        return {
            **state,
            "final_answer": (
                "No policy documents found in the cloud bucket. "
                "Please run the PDF extractor first."
            ),
        }

    # Stream each .md file and build the combined context
    combined_context = ""
    loaded_count = 0

    for md_filename in sorted(md_files):
        # Skip the cache file if it accidentally ends with .md
        if md_filename == CACHE_KEY:
            continue
        try:
            obj_res = s3_client.get_object(Bucket=BUCKET_NAME, Key=md_filename)
            content = obj_res["Body"].read().decode("utf-8")
            combined_context += f"\n\n=== {md_filename} ===\n{content}"
            loaded_count += 1
        except ClientError as e:
            print(f"  WARNING: Could not fetch {md_filename}: {e}")
            continue

    print(f"  Loaded {loaded_count} policy documents ({len(combined_context)} chars total).")

    if not combined_context.strip():
        return {
            **state,
            "final_answer": (
                "Policy documents could not be loaded from the cloud. "
                "Please check your R2 connection and try again."
            ),
        }

    policy_prompt = POLICY_PROMPT_TEMPLATE.format(
        combined_context=combined_context,
        user_query=user_query,
    )

    response = llm.invoke([HumanMessage(content=policy_prompt)])
    answer = _strip_thinking(response.content).strip()
    print(f"  Answer generated ({len(answer)} characters).")
    return {**state, "final_answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — GENERAL AGENT NODE
# ─────────────────────────────────────────────────────────────────────────────

GENERAL_PROMPT_TEMPLATE = """
You are a friendly and helpful virtual assistant at Nova Technologies Private Limited.
The employee has asked something that is not related to company policies.

Respond in a warm, helpful, and concise way. Keep your answer under 100 words.
If they are greeting you, greet them back warmly.

Employee message: {user_query}
"""


def general_agent_node(state: AgentState) -> AgentState:
    """
    Handles casual / non-policy queries without touching R2.
    """
    print("\n[NODE 5] General Agent — answering as a friendly assistant...")
    user_query = state.get("user_query", "")

    general_prompt = GENERAL_PROMPT_TEMPLATE.format(user_query=user_query)

    response = llm.invoke([HumanMessage(content=general_prompt)])
    answer = _strip_thinking(response.content).strip()
    print(f"  Response: {answer[:80]}...")
    return {**state, "final_answer": answer}