# civic-backend

FastAPI backend for the Smart City Civic Issue Reporting System — Bengaluru Municipal Corporation.

Handles SQLite persistence, RAG (retrieval-augmented generation), LangGraph conversation workflows, and serves the React frontend as static files — all on a single port.

---

## Project Structure

```
civic-backend/
├── main.py          # FastAPI app — all endpoints, startup, static file serving
├── graph.py         # LangGraph state machine (intake → categorize → priority → done)
├── rag_setup.py     # RAG with HuggingFace embeddings + FAISS vector store
├── models.py        # Pydantic models — Complaint, Priority, Status, Category
├── database.py      # SQLite persistence layer
├── requirements.txt # Python dependencies
├── .env             # LLM provider config (copy from .env.example)
├── .env.example     # Template for environment variables
├── civic.db         # SQLite database (auto-created on first run)
└── dist/            # Built React frontend (copied from civic-frontend)
```

---

## Setup (First Time)

```cmd
cd C:\Users\Dhruva\Documents\civic-backend
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values:
```cmd
copy .env.example .env
```

---

## Running

```cmd
cd C:\Users\Dhruva\Documents\civic-backend
python main.py
```

The browser opens automatically at `http://127.0.0.1:8000`.

To stop: `Ctrl+C` in the terminal.

> Note: `reload=True` is disabled when running via `python main.py`. After any backend code change, stop and rerun manually.

---

## Environment Variables (`.env`)

```env
LLM_PROVIDER=ollama          # or groq
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
GROQ_API_KEY=gsk_...         # only needed if using Groq
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## API Endpoints

### Chat
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | LangGraph conversation endpoint (session-aware) |

### Complaints
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/complaints` | Save a new complaint from the frontend |
| GET | `/api/status/{id}` | Get status of a complaint by ID |

### Officer Dashboard
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/officer/dashboard` | Load all complaints (stats, recent, hot zones) |
| PUT | `/api/officer/update/{id}` | Update status or similar_count |
| POST | `/api/officer/escalate` | Escalate a complaint |

### Frontend
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the React app (index.html) |
| GET | `/assets/*` | Serves JS/CSS bundles |

---

## Database

SQLite file: `civic.db` — auto-created in the backend folder on first run.

### Tables
- `complaints` — all complaint records (18 fields)
- `conversations` — chat history per session

### Useful commands

**Test the database:**
```cmd
python database.py
```

**Clear all data and start fresh:**
```cmd
del civic.db
```
Then restart — a new empty database is created automatically.

**Or clear without deleting the file:**
```cmd
python -c "import sqlite3; conn = sqlite3.connect('civic.db'); conn.execute('DELETE FROM complaints'); conn.execute('DELETE FROM conversations'); conn.commit(); conn.close(); print('Cleared')"
```

---

## LangGraph Workflow

```
User message
     │
     ▼
 intake_node          → detect intent (report / track / info / exit)
     │
     ▼
 categorize_node      → extract category, zone, description
     │
     ▼
 duplicate_check_node → RAG similarity search against existing complaints
     │
     ├─ duplicate found → return linked complaint ID
     │
     ▼
 priority_assign_node → compute priority, assign dept, save to SQLite ← HERE
     │
     ▼
 response
```

The SQLite save happens **inside `priority_assign_node`** to avoid LangGraph state serialization issues where Pydantic models are lost between nodes.

---

## RAG (Retrieval-Augmented Generation)

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
- Vector store: FAISS (in-memory, rebuilt on each startup from SQLite)
- Knowledge base: BBMP policies, SLA rules, zone-based priority logic
- Used for: duplicate detection and context injection into LLM prompts

---

## Dependencies

Key packages:
```
fastapi
uvicorn
langchain
langchain-groq
langchain-ollama
langchain-huggingface
langgraph
faiss-cpu
sentence-transformers
pydantic
python-dotenv
sqlite3 (built-in)
```

Install all:
```cmd
pip install -r requirements.txt
```

---

## Startup Log (expected)

```
[DB] SQLite initialized at: C:\...\civic-backend\civic.db
[RAG] Initializing embeddings...
[RAG] Indexed 15 policy chunks
[RAG] Ready ✓
[DB] Loaded 3 existing complaints from SQLite
[RAG] Re-indexed 3 complaints
[STATIC] Looking for dist at: C:\...\civic-backend\dist
[STATIC] dist exists: True
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## Architecture

```
Browser (React)
      │
      │  HTTP / REST
      ▼
FastAPI (main.py) — port 8000
      │
      ├── /api/complaints   ──► database.py (SQLite)
      ├── /api/chat         ──► graph.py (LangGraph) ──► LLM (Ollama / Groq)
      ├── /api/officer/*    ──► database.py (SQLite)
      └── /*                ──► dist/ (React static files)
```
