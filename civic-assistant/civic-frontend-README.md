# civic-frontend

React frontend for the Smart City Civic Issue Reporting System — Bengaluru Municipal Corporation.

Built with **React + Vite**. Communicates with the FastAPI backend at `http://127.0.0.1:8000`.

---

## Project Structure

```
civic-frontend/
├── src/
│   ├── SmartCityCivicAssistant.jsx   # Main app — all UI and logic lives here
│   ├── App.jsx                        # Root wrapper
│   └── index.css                      # Empty (all styles are inline)
├── dist/                              # Built output (copy this to civic-backend/)
├── index.html
├── vite.config.js
└── package.json
```

---

## Setup (First Time)

```cmd
cd C:\Users\Dhruva\Downloads
npm create vite@latest civic-frontend -- --template react
cd civic-frontend
npm install
```

Then copy `SmartCityCivicAssistant.jsx`, `App.jsx`, and `index.css` into `src/`.

---

## Running

### Development (live reload)
```cmd
cd C:\Users\Dhruva\Downloads\civic-frontend
npm run dev
```
Opens at `http://localhost:5173`. The backend must be running on port 8000 for API calls to work.

### Production (served via backend)
```cmd
npm run build
xcopy "C:\Users\Dhruva\Downloads\civic-frontend\dist" "C:\Users\Dhruva\Downloads\civic-backend\dist" /E /I /Y
```
Then run the backend — the app is served at `http://127.0.0.1:8000`.

**After every JSX change:** rebuild, xcopy, then hard refresh with `Ctrl+Shift+R`.

---

## Features

### Citizen View
- Conversational chatbot powered by **Ollama** (local) or **Groq** (cloud) — switchable in the header
- LangGraph-style state machine: `intake → categorize → duplicate_check → confirm → done`
- Automatic duplicate detection — links report to existing complaint and increments priority
- Complaint registration with auto-assigned priority, department, and SLA
- Complaints saved to backend SQLite on registration
- Track complaints by ID via chat or the sidebar
- Chat ends and input is disabled when exit intent is detected ("thank you", "bye", etc.)

### Officer Dashboard
- Stats: Total, Open, In Progress, Escalated, Resolved
- Priority queue table with live status updates (synced to backend)
- Hot zones heatmap
- Category breakdown chart
- RAG Intelligence info panel

### Short-circuit Handlers (no LLM call)
| User says | Action |
|---|---|
| "thank you / bye / that's it" | Ends conversation, disables input |
| "Track #ID" | Looks up complaint locally |
| "Go to Officer Dashboard" | Switches tab instantly |
| "Report another issue" | Resets state, prompts for new report |

---

## LLM Providers

| Provider | Model | How |
|---|---|---|
| Ollama | llama3.2:3b | Runs locally — `ollama serve` must be running |
| Groq | llama-3.3-70b-versatile | Cloud API — enter free key at [console.groq.com](https://console.groq.com) |

---

## Backend API Calls

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/officer/dashboard` | Load existing complaints on startup |
| POST | `/api/complaints` | Save new complaint to SQLite |
| PUT | `/api/officer/update/:id` | Update status or similar_count |

---

## Dependencies

```json
"react": "^18",
"react-dom": "^18",
"vite": "^5"
```

No extra npm packages required — LLM calls go directly to Ollama/Groq via `fetch()`.
