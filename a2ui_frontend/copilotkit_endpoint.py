"""
copilotkit_endpoint.py  —  Backend Server with R2 Cloud Storage Integration

Plain FastAPI backend. No CopilotKit SDK needed.
Receives questions from the React frontend and forwards them to the A2A server.
Also handles file uploads to Cloudflare R2 and serves PDFs back via API
so the pipeline no longer depends on local disk storage.

RUN:  python copilotkit_endpoint.py
"""

import os
import re
import io
import requests
import uvicorn
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://localhost:8000")

app = FastAPI(title="Policy Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

BUCKET_NAME          = os.getenv('R2_BUCKET_NAME', 'nova-policy-bucket')
ALLOWED_EXTENSIONS    = {".pdf", ".zip"}
MAX_FILE_SIZE_MB       = 20


class QueryRequest(BaseModel):
    question: str


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Sanitize filenames before storing in R2
#
# Strips path components and replaces unsafe characters so a malicious
# filename (e.g. "../../config.json") cannot affect the object key.
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename)               # strip any path parts
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename) # allow only safe chars
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
        file_obj       = io.BytesIO(contents)

        s3_client.upload_fileobj(file_obj, BUCKET_NAME, safe_filename)

        return {
            "status": "success",
            "message": f"Successfully uploaded {safe_filename} to {BUCKET_NAME}.",
            "filename": safe_filename,
        }

    except Exception as e:
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
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/query")
async def query(req: QueryRequest):
    """Forwards user question to the LangGraph A2A server."""
    try:
        response = requests.post(
            f"{A2A_SERVER_URL}/a2a",
            json={
                "id": "frontend-query",
                "message": {"role": "user", "content": req.question},
            },
            timeout=300,
        )
        data = response.json()
        return {
            "answer":     data["message"]["content"],
            "query_type": data.get("metadata", {}).get("query_type", "unknown"),
        }
    except requests.exceptions.ConnectionError:
        return {"answer": "Cannot connect to the A2A server. Is it running on port 8000?", "query_type": "error"}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "query_type": "error"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("\n" + "="*45)
    print("  Backend → http://localhost:8001")
    print("  A2A     → " + A2A_SERVER_URL)
    print("  R2 Bucket → " + BUCKET_NAME)
    print("="*45 + "\n")
    uvicorn.run("copilotkit_endpoint:app", host="0.0.0.0", port=8001, reload=True)