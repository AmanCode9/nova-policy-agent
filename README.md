# Nova Technologies Policy Agent

A multi-agent, LLM-powered Q&A system that lets employees ask natural-language questions about company policies — leave, travel, IT assets, code of conduct, onboarding — and get accurate, cited answers in seconds, instead of digging through PDFs.

Built during a Generative AI internship at **TCG Digital** as a client-facing deliverable for **Nova Technologies**.

---

## Why this exists

Company policy documents are long, scattered across PDFs, and rarely read end-to-end by employees. This project turns five policy documents into a conversational agent that:

- Understands the intent behind a question (policy-related vs. general chat) and routes it accordingly
- Retrieves the right policy content instead of guessing
- Answers only from source documents, and says so explicitly when it can't find an answer
- Is continuously graded on faithfulness and correctness through an automated evaluation pipeline — not just "does it respond," but "is the response actually right"

---

## Architecture

```
┌─────────────┐      HTTP       ┌──────────────────────┐
│   React UI   │ ───────────────▶│   FastAPI Backend     │
│  (frontend)  │◀─────────────── │ (copilotkit_endpoint) │
└─────────────┘                  └───────────┬───────────┘
                                              │ invokes
                                              ▼
                                  ┌───────────────────────┐
                                  │   LangGraph Pipeline    │
                                  │                         │
                                  │  cache_check             │
                                  │      │                   │
                                  │      ├─ new PDFs? ─┐      │
                                  │      │             ▼      │
                                  │      │      pdf_extractor  │
                                  │      │             │       │
                                  │      ▼◀────────────┘       │
                                  │  orchestrator (classify)   │
                                  │      │                     │
                                  │      ├─ policy → policy_agent│
                                  │      └─ general → general_agent│
                                  └───────────┬─────────────────┘
                                              │
                                              ▼
                                  ┌───────────────────────┐
                                  │  Cloudflare R2 (S3 API) │
                                  │  PDFs · Markdown · cache│
                                  └───────────────────────┘

                 Powered end-to-end by Google Gemini (gemini-2.0-flash)
```

**Flow:** a user asks a question in the React UI → the FastAPI backend passes it to a LangGraph pipeline → the pipeline checks whether new policy PDFs need processing, classifies the query, retrieves the relevant policy markdown from cloud storage, and generates a grounded answer with Gemini.

---

## Key features

- **Cloud-native document store** — Policy PDFs, their extracted Markdown, and the processing cache all live in Cloudflare R2 (S3-compatible), not on local disk. The pipeline lists, reads, and writes documents entirely through the R2 API.
- **Automatic PDF → Markdown pipeline** — New policy PDFs are automatically detected, converted into structured, agent-readable Markdown by an LLM extraction step, and cached so they're never reprocessed twice.
- **Intent-based routing** — An orchestrator node classifies each query as *policy* or *general* and routes it to a specialized agent, so casual chat doesn't waste a full document-retrieval pass.
- **Grounded, source-cited answers** — The policy agent is instructed to answer only from retrieved documents and explicitly say when something isn't covered, rather than hallucinating.
- **MCP tool servers** — Policy file listing, reading, keyword search, and cache status are exposed as standalone MCP (Model Context Protocol) tools, independently tested and ready for agent-tool integration.
- **Automated evaluation suite** — Every pipeline response is benchmarked with DeepEval across faithfulness, contextual relevancy, and answer correctness, using an LLM-as-judge setup — so answer quality is measured, not assumed.

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM & reasoning | Google Gemini (`gemini-2.0-flash`) via `langchain-google-genai` |
| Agent orchestration | LangGraph |
| Tool layer | MCP (Model Context Protocol) |
| Backend API | FastAPI, Uvicorn |
| Cloud storage | Cloudflare R2 (S3-compatible) via `boto3` |
| Frontend | React |
| Evaluation | DeepEval (LLM-as-judge), pytest |
| PDF parsing | pypdf |

---

## Project structure

```
├── nodes.py                 # LangGraph node functions (cache check, extraction, routing, answering)
├── graph.py                 # Builds and compiles the LangGraph pipeline
├── state.py                 # Shared pipeline state schema
├── md_reader.py              # MCP tool server — read/search policy markdown in R2
├── cache_checker.py          # MCP tool server — track which PDFs are processed
├── copilotkit_endpoint.py    # FastAPI backend — file upload + query API for the frontend
├── App.js / App.css / index.js  # React chat interface
├── eval_dataset.py           # Test cases + retrieval-context loader for evaluation
├── test_policy_agent.py      # DeepEval suite — faithfulness, relevancy, correctness
├── test_mcp_tools.py         # Unit tests for MCP tool servers
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Google Gemini API key
- A Cloudflare R2 bucket + access credentials

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd nova-policy-agent

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
npm install
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
R2_ACCOUNT_ID=your_r2_account_id
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=nova-policy-bucket
```

> Never commit `.env` or hardcode API keys in source files — see `.gitignore`.

### 3. Run the backend

```bash
python copilotkit_endpoint.py
```
Starts the API server at `http://localhost:8001`.

### 4. Run the frontend

```bash
npm start
```
Opens the chat UI at `http://localhost:3000`.

### 5. Try it out

Upload a policy PDF through the UI, or drop one directly into your R2 bucket, then ask questions like:

- *"How many casual leaves do I get per year?"*
- *"What's the travel allowance for a manager?"*
- *"What do I do if I lose my company laptop?"*

---

## Evaluation

Answer quality is measured automatically rather than eyeballed. Run the full evaluation suite with:

```bash
pytest test_mcp_tools.py -v       # MCP tool correctness (unit-level)
deepeval test run test_policy_agent.py   # End-to-end answer quality (LLM-as-judge)
```

Metrics tracked per query:
- **Faithfulness** — does the answer stay true to the retrieved policy document?
- **Contextual relevancy** — was the right document actually retrieved?
- **Answer correctness / policy accuracy** — is the final answer factually right?

---

## Roadmap

- [ ] Expand evaluation coverage across all five policy documents
- [ ] Wire MCP tool servers directly into the LangGraph pipeline (currently validated independently)
- [ ] Add lightweight API-key auth to backend endpoints
- [ ] Structured logging in place of `print()` statements
- [ ] Deploy backend + frontend to a public environment

---

## Author

Built by **Aman Patra**, B.Tech Computer Science, KIIT Bhubaneswar — as part of a Generative AI internship at TCG Digital.