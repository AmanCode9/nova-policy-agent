"""
cache_checker.py  —  MCP Tool #1: Cache Checker (R2 Cloud Storage Version)

This MCP server exposes tools that the LangGraph pipeline calls to decide
whether the PDF Extractor Agent needs to run or can be skipped.

WHAT CHANGED FROM THE LOCAL VERSION:
  - PDF listing no longer uses os.listdir(PDF_FOLDER).
    Instead, it calls s3_client.list_objects_v2() — a real API call
    to Cloudflare R2's S3-compatible REST API. This is the "PDF files
    API call" requirement.

  - cache.json is no longer read/written to local disk.
    Instead, it is stored as an object inside the R2 bucket
    (key: "cache.json") using s3_client.get_object() / put_object().
    This is the "cache will be saved in the bucket" requirement.

HOW THE CACHE WORKS:
  - cache.json (inside the bucket) tracks every PDF filename that has
    already been processed into a .md file.
  - On every run, we list PDFs currently in the bucket via the R2 API
    and compare against the cache object (also in the bucket).
  - If all PDFs are already in the cache → return "no" (skip extraction)
  - If any PDF is new or missing from cache → return "yes" (run extraction)

Run this file to start the MCP server:
    python cache_checker.py
"""

import os
import json
import logging
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Path configuration ──────────────────────────────────────────────────────
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
CACHE_KEY   = "cache.json"   # the object key for the cache file INSIDE the bucket

# ── Create the MCP server ────────────────────────────────────────────────────
mcp = FastMCP("cache-checker")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: List all PDFs currently in the R2 bucket
#
# This replaces os.listdir(PDF_FOLDER). It calls list_objects_v2(),
# which is an HTTPS API request to R2's S3-compatible endpoint.
# ─────────────────────────────────────────────────────────────────────────────
def list_bucket_pdfs() -> list:
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        contents = response.get("Contents", [])
        return [obj["Key"] for obj in contents if obj["Key"].lower().endswith(".pdf")]
    except ClientError:
        logger.exception("R2 list_objects_v2 error")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Load the cache object from the R2 bucket
#
# This replaces open(CACHE_FILE, "r"). It calls get_object(), which is
# an HTTPS API request to fetch the cache.json object from the bucket.
# ─────────────────────────────────────────────────────────────────────────────
def load_cache() -> dict:
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=CACHE_KEY)
        content  = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "NoSuchKey":
            # First time running — no cache object yet in the bucket
            return {"processed_files": []}
        logger.exception("R2 get_object error while loading cache")
        return {"processed_files": []}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Save the cache object back to the R2 bucket
#
# This replaces open(CACHE_FILE, "w"). It calls put_object(), which is
# an HTTPS API request that uploads the updated cache.json to the bucket.
# ─────────────────────────────────────────────────────────────────────────────
def save_cache(cache: dict):
    try:
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=CACHE_KEY,
            Body=json.dumps(cache, indent=2),
            ContentType="application/json",
        )
    except ClientError:
        logger.exception("R2 put_object error while saving cache")


# ── MCP Tool 1: check_cache ──────────────────────────────────────────────────
@mcp.tool()
def check_cache() -> str:
    """
    Compares the PDFs currently in the R2 bucket with the processed files
    listed in the cache object (also stored in the bucket).

    Returns:
        "yes"  — one or more PDFs have not been processed yet → run the extractor
        "no"   — all PDFs are already processed → skip extraction, go straight to Q&A
    """
    all_pdfs = list_bucket_pdfs()

    if not all_pdfs:
        return "no"   # Bucket has no PDFs to process

    cache = load_cache()
    already_processed = set(cache.get("processed_files", []))

    new_pdfs = [pdf for pdf in all_pdfs if pdf not in already_processed]

    if new_pdfs:
        logger.info("New PDFs found in bucket that need processing: %s", new_pdfs)
        return "yes"
    else:
        logger.info("All %d PDFs in bucket already processed. Skipping extraction.", len(all_pdfs))
        return "no"


# ── MCP Tool 2: mark_as_processed ────────────────────────────────────────────
@mcp.tool()
def mark_as_processed(filename: str) -> str:
    """
    Adds a PDF filename to the cache (stored in the R2 bucket) after it has
    been successfully processed.

    Args:
        filename: The PDF filename to mark as done, e.g. "hr_leave_policy.pdf"

    Returns:
        A confirmation message string.
    """
    cache = load_cache()

    if filename not in cache["processed_files"]:
        cache["processed_files"].append(filename)
        save_cache(cache)
        logger.info("Marked as processed: %s", filename)
        return f"OK: '{filename}' added to cache (saved in bucket)."
    else:
        return f"INFO: '{filename}' was already in the cache."


# ── MCP Tool 3: get_cache_status ─────────────────────────────────────────────
@mcp.tool()
def get_cache_status() -> str:
    """
    Returns a human-readable summary of the current cache status,
    based on PDFs in the R2 bucket and the cache object in the bucket.

    Returns:
        A formatted string showing total PDFs, processed count, and pending list.
    """
    all_pdfs  = list_bucket_pdfs()
    cache     = load_cache()
    processed = cache.get("processed_files", [])
    pending   = [f for f in all_pdfs if f not in processed]

    status = (
        f"Bucket            : {BUCKET_NAME}\n"
        f"Total PDFs in bucket : {len(all_pdfs)}\n"
        f"Already processed    : {len(processed)}\n"
        f"Pending (not cached) : {len(pending)}\n"
        f"\nProcessed files:\n"
        + "\n".join(f"  [DONE] {f}" for f in processed)
        + ("\n\nPending files:\n" if pending else "")
        + "\n".join(f"  [TODO] {f}" for f in pending)
    )
    return status


# ── MCP Tool 4: reset_cache ───────────────────────────────────────────────────
@mcp.tool()
def reset_cache() -> str:
    """
    Clears the entire cache object stored in the R2 bucket. Use this if you
    want to reprocess all PDFs from scratch.

    Returns:
        A confirmation message string.
    """
    save_cache({"processed_files": []})
    logger.info("Cache (in bucket) has been reset.")
    return "Cache cleared in bucket. All PDFs will be reprocessed on the next run."


# ── MCP Tool 5: get_unprocessed_pdfs ─────────────────────────────────────────
@mcp.tool()
def get_unprocessed_pdfs() -> str:
    """
    Returns the list of PDF filenames in the R2 bucket that have NOT been
    processed yet. The PDF Extractor Agent uses this to know exactly which
    files to fetch and read.

    Returns:
        A comma-separated list of filenames, or "none" if everything is processed.
    """
    all_pdfs  = list_bucket_pdfs()
    cache     = load_cache()
    processed = set(cache.get("processed_files", []))
    pending   = [f for f in all_pdfs if f not in processed]

    if not pending:
        return "none"

    return ",".join(pending)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Cache Checker MCP server (R2 Cloud Storage version)...")
    logger.info("R2 Bucket   : %s", BUCKET_NAME)
    logger.info("Cache key   : %s (stored inside bucket)", CACHE_KEY)
    logger.info("Available tools: check_cache, mark_as_processed, get_cache_status, reset_cache, get_unprocessed_pdfs")
    mcp.run(transport="stdio")