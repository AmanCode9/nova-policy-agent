"""
md_reader.py  —  MCP Tool #2: MD Reader (R2 Cloud Storage Version)

This MCP server exposes tools for the LangGraph agents to read, search,
and summarize the extracted markdown (.md) policy files.

WHAT CHANGED FROM THE LOCAL VERSION:
  - No longer uses os.listdir() or open() from a local "md_outputs" folder.
  - Connects to the Cloudflare R2 bucket using boto3.
  - Lists and streams the .md files directly from the cloud.
  - Uses pagination to safely handle buckets with more than 1,000 files.
"""

import os
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

# ── Create the MCP server ────────────────────────────────────────────────────
mcp = FastMCP("md-reader")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: List all .md files currently in the R2 bucket (with Pagination)
# ─────────────────────────────────────────────────────────────────────────────
def get_all_md_files() -> list:
    """Returns a list of all .md object keys in the bucket."""
    md_files = []
    try:
        # Paginator ensures we get all files even if there are > 1,000 in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME)

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj["Key"].lower().endswith(".md"):
                        md_files.append(obj["Key"])
        return md_files
    except ClientError:
        logger.exception("R2 list_objects_v2 error")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Fetch file content from the R2 bucket
# ─────────────────────────────────────────────────────────────────────────────
def get_file_content(filename: str) -> str:
    """Fetches and decodes the content of a specific file from the bucket."""
    if not filename.endswith(".md"):
        filename = filename + ".md"

    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=filename)
        return response["Body"].read().decode("utf-8")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "NoSuchKey":
            return None  # File doesn't exist
        logger.exception("R2 get_object error for %s", filename)
        return None


# ── MCP Tool 1: list_policy_files ────────────────────────────────────────────
@mcp.tool()
def list_policy_files() -> str:
    """
    Lists all the .md policy files available in the R2 bucket.
    The orchestrator agent can call this to know what documents are available.
    """
    md_files = get_all_md_files()

    if not md_files:
        return f"No .md files found in the bucket '{BUCKET_NAME}'. Run the PDF extractor first."

    result = f"Available policy documents ({len(md_files)} total):\n"
    for f in sorted(md_files):
        result += f"  - {f}\n"
    return result


# ── MCP Tool 2: read_policy_file ─────────────────────────────────────────────
@mcp.tool()
def read_policy_file(filename: str) -> str:
    """
    Reads and returns the full content of a specific .md policy file from R2.
    The Policy Agent uses this to fetch a document's content before answering.
    """
    content = get_file_content(filename)

    if content is None:
        md_files = get_all_md_files()
        available = md_files if md_files else "none"
        return f"File '{filename}' not found in bucket. Available files: {available}"

    logger.info("Loaded: %s from R2 (%d characters)", filename, len(content))
    return content


# ── MCP Tool 3: search_policies ──────────────────────────────────────────────
@mcp.tool()
def search_policies(keyword: str) -> str:
    """
    Searches all .md policy files in R2 for a given keyword and returns matching excerpts.
    This is a simple keyword search. The Policy Agent can use this to quickly
    find which document to dig deeper into.
    """
    md_files = get_all_md_files()

    if not md_files:
        return "No .md files found in the bucket. Run the PDF extractor first."

    keyword_lower = keyword.lower()
    results = []

    # Stream and search each file
    for filename in sorted(md_files):
        content = get_file_content(filename)
        if not content:
            continue

        lines = content.split("\n")

        # Find lines containing the keyword and show some context around them
        for i, line in enumerate(lines):
            if keyword_lower in line.lower():
                # Grab up to 3 lines of context around the matching line
                start = max(0, i - 1)
                end   = min(len(lines), i + 3)
                snippet = "\n".join(lines[start:end]).strip()

                results.append(
                    f"\n--- From: {filename} ---\n{snippet}\n"
                )

    if not results:
        return f"No results found for keyword: '{keyword}'"

    return f"Search results for '{keyword}':\n" + "\n".join(results)


# ── MCP Tool 4: get_policy_summary ───────────────────────────────────────────
@mcp.tool()
def get_policy_summary(filename: str) -> str:
    """
    Returns only the Overview and Scope sections of a policy document.
    Useful when the agent needs a quick summary without loading the full document.
    """
    content = get_file_content(filename)

    if content is None:
        return f"File '{filename}' not found in the bucket."

    # Extract lines under "Overview" or "Scope" sections
    lines         = content.split("\n")
    capture       = False
    summary_lines = []
    section_count = 0

    for line in lines:
        # Look for Overview or Scope headings (markdown ## or ###)
        if any(kw in line for kw in ["Overview", "Scope", "Purpose"]):
            capture = True
            section_count += 1
            summary_lines.append(line)
            continue

        # Stop after 2 sections worth of content
        if capture and line.startswith("#") and section_count >= 2:
            break

        if capture:
            summary_lines.append(line)

    if summary_lines:
        return "\n".join(summary_lines)
    else:
        # Fall back: return first 500 characters
        return content[:500] + "..."


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting MD Reader MCP server (R2 Cloud Storage version)...")
    logger.info("R2 Bucket: %s", BUCKET_NAME)
    logger.info("Available tools: list_policy_files, read_policy_file, search_policies, get_policy_summary")
    mcp.run(transport="stdio")