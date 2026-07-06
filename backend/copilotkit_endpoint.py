"""
copilotkit_endpoint.py  —  Backend Server with R2 Cloud Storage Integration

Plain FastAPI backend. No CopilotKit SDK needed.
Receives questions from the React frontend and runs them directly through the
LangGraph policy Q&A pipeline (no A2A hop). Also handles file uploads to
Cloudflare R2 and serves PDFs back via API so the pipeline no longer depends
on local disk storage.

RUN:  python copilotkit_endpoint.py
"""

import os
import re
import io
import logging
import asyncio
import sys

import uvicorn
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

PIPELINE_DIR = os.path.join(PROJECT_DIR, "langgraph_pipeline")     
sys.path.insert(0, PIPELINE_DIR)

from graph import build_graph   

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app = FastAPI(title="Policy Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH PIPELINE — built once at startup, reused across every request
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Building LangGraph pipeline...")
graph_app = build_graph()
logger.info("Pipeline ready.")

# ─────────────────────────────────────────────────────────────────────────────
# CLOUDFLARE R2 S3 CLIENT INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
s3_client = boto3.client(
    service_name='s3',
    endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    region_name='auto',
)

BUCKET_NAME        = os.getenv('R2_BUCKET_NAME', 'nova-policy-bucket')
ALLOWED_EXTENSIONS = {".pdf", ".zip"}
MAX_FILE_SIZE_MB   = 20


class QueryRequest(BaseModel):
    question: str


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Sanitize filenames before storing in R2
#
# Strips path components and replaces unsafe characters so a malicious
# filename (e.g. "../../config.json") cannot affect the object key.
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename)                # strip any path parts
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)  # allow only safe chars
    return filename


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — FILE UPLOAD
# Receives a file from the React UI and uploads it to Cloudflare R2.
# Validates file type and size before uploading.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Validates and uploads a PDF/ZIP file to the R2 bucket."""
    try:
        contents = await file.read()
        size_mb  = len(contents) / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:
            return {
                "status": "error",
                "message": f"File too large ({size_mb:.1f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB.",
            }

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return {
                "status": "error",
                "message": f"File type '{ext}' not allowed. Only PDF and ZIP files are accepted.",
            }

        safe_filename = sanitize_filename(file.filename)
        file_obj      = io.BytesIO(contents)

        s3_client.upload_fileobj(file_obj, BUCKET_NAME, safe_filename)

        logger.info("Uploaded %s to bucket %s", safe_filename, BUCKET_NAME)

        return {
            "status": "success",
            "message": f"Successfully uploaded {safe_filename} to {BUCKET_NAME}.",
            "filename": safe_filename,
        }

    except Exception as e:
        logger.exception("Upload failed")
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — LIST PDF FILES IN BUCKET
#
# Used by the pipeline (pdf_extractor_node) instead of os.listdir(PDF_FOLDER).
# Returns only files ending in .pdf so the agent knows what's available
# without needing local disk access.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/pdfs")
async def list_pdfs():
    """Lists all PDF filenames currently stored in the R2 bucket."""
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        contents = response.get("Contents", [])

        pdf_files = [
            obj["Key"] for obj in contents
            if obj["Key"].lower().endswith(".pdf")
        ]

        return {"status": "success", "pdfs": pdf_files, "count": len(pdf_files)}

    except ClientError as e:
        logger.exception("Failed to list PDFs from bucket")
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — FETCH A SPECIFIC PDF FROM BUCKET
#
# Used by the pipeline (pdf_extractor_node) instead of open(filepath).
# Streams the PDF bytes back so they can be passed directly to pypdf
# or any other reader, without ever touching local disk.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/pdfs/{filename}")
async def get_pdf(filename: str):
    """Downloads a specific PDF file's bytes from the R2 bucket."""
    try:
        safe_filename = sanitize_filename(filename)
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=safe_filename)

        file_stream = response["Body"]

        return StreamingResponse(
            file_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
        )

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "NoSuchKey":
            raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found in bucket.")
        logger.exception("Failed to fetch PDF %s from bucket", filename)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — QUERY THE POLICY AGENT
#
# Runs the user's question directly through the in-process LangGraph
# pipeline (no A2A/HTTP hop). graph_app.invoke() is a blocking, synchronous
# call, so it's offloaded to a worker thread via run_in_executor — otherwise
# a single slow Gemini call would stall the entire async event loop and
# block every other request the backend is trying to serve.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/query")
async def query(req: QueryRequest):
    """Runs the user's question through the LangGraph pipeline and returns the answer."""
    try:
        loop = asyncio.get_event_loop()

        initial_state = {
            "user_query":        req.question,
            "extraction_needed": "",
            "extraction_done":   False,
            "query_type":        "",
            "final_answer":      "",
            "error":             None,
        }

        result = await loop.run_in_executor(None, graph_app.invoke, initial_state)

        answer     = result.get("final_answer", "I could not generate an answer.")
        query_type = result.get("query_type", "unknown")

        return {"answer": answer, "query_type": query_type}

    except Exception as e:
        logger.exception("Pipeline execution failed for query: %s", req.question)
        return {"answer": f"Error: {str(e)}", "query_type": "error"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    logger.info("Backend starting at http://localhost:8001")
    logger.info("R2 Bucket: %s", BUCKET_NAME)
    uvicorn.run("copilotkit_endpoint:app", host="0.0.0.0", port=8001, reload=True)