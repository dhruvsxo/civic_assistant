"""
Smart City Civic Issue Assistant - FastAPI Backend
==================================================
Handles citizen complaints via conversational AI using LangGraph + LangChain + RAG.
Uses SQLite for persistent storage via database.py
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone
from pathlib import Path
import os

from graph import build_complaint_graph, ComplaintState
from rag_setup import CivicRAG
from models import (
    Complaint, ComplaintStatus, ComplaintCategory,
    Priority, OfficerDashboard
)
from database import (
    init_db, save_complaint, load_complaint,
    load_all_complaints, update_complaint_field,
    save_message, load_conversation
)

app = FastAPI(
    title="Smart City Civic Issue Assistant",
    description="AI-powered civic issue reporting and resolution system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB and RAG on startup
init_db()
rag = CivicRAG()
complaint_graph = build_complaint_graph(rag)

# Load existing complaints into memory + re-index in RAG
complaints_db: dict[str, Complaint] = load_all_complaints()
conversations: dict[str, list] = {}
print(f"[DB] Loaded {len(complaints_db)} existing complaints from SQLite")
for complaint in complaints_db.values():
    rag.index_complaint(complaint)
print(f"[RAG] Re-indexed {len(complaints_db)} complaints")


# ── Schemas ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    session_id: str
    message: str
    location: Optional[dict] = None

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    complaint_id: Optional[str] = None
    state: Optional[str] = None
    suggestions: Optional[List[str]] = None

class EscalateRequest(BaseModel):
    complaint_id: str
    reason: str
    officer_id: str

class ComplaintCreate(BaseModel):
    id: str
    category: str
    priority: str
    status: str = "OPEN"
    zone: str
    description: str
    assigned_to: str
    similar_count: int = 0
    duplicate_of: Optional[str] = None
    session_id: str = ""
    location: Optional[dict] = None


# ── Save Complaint from Frontend ────────────────────────────────────────────────

@app.post("/api/complaints")
async def create_complaint(data: ComplaintCreate):
    """Called by the React frontend whenever a complaint is registered."""
    print(f"\n[COMPLAINT] ── New complaint received ──────────────")
    print(f"[COMPLAINT] ID       : {data.id}")
    print(f"[COMPLAINT] Category : {data.category}")
    print(f"[COMPLAINT] Priority : {data.priority}")
    print(f"[COMPLAINT] Zone     : {data.zone}")
    print(f"[COMPLAINT] Desc     : {data.description[:80]}")
    print(f"[COMPLAINT] ────────────────────────────────────────\n")

    try:
        complaint = Complaint(
            id=data.id,
            session_id=data.session_id,
            description=data.description,
            category=ComplaintCategory(data.category),
            priority=Priority(data.priority),
            status=ComplaintStatus(data.status),
            zone=data.zone,
            assigned_to=data.assigned_to,
            similar_count=data.similar_count,
            duplicate_of=data.duplicate_of,
            location=data.location,
        )
        complaints_db[complaint.id] = complaint
        save_complaint(complaint)
        print(f"[COMPLAINT] ✓ Saved {complaint.id} to SQLite")
        return {"success": True, "complaint_id": complaint.id}
    except Exception as e:
        print(f"[COMPLAINT] ✗ Failed to save: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Citizen Chat Endpoint ───────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage, background_tasks: BackgroundTasks):
    session_id = msg.session_id

    # Load conversation history from SQLite
    history = load_conversation(session_id)
    history.append({"role": "user", "content": msg.message})
    save_message(session_id, "user", msg.message)

    state = ComplaintState(
        session_id=session_id,
        messages=history,
        location=msg.location,
        complaints_db=complaints_db,
        rag=rag,
    )

    result = await complaint_graph.ainvoke(state)
    reply = result["reply"]
    save_message(session_id, "assistant", reply)

    # Debug — print everything the graph returned
    print(f"\n[CHAT] ── Graph result ──────────────────────────")
    print(f"[CHAT] graph_state   : {result.get('graph_state')}")
    print(f"[CHAT] complaint_id  : {result.get('complaint_id')}")
    print(f"[CHAT] complaint_obj : {result.get('complaint_obj')}")
    print(f"[CHAT] reply preview : {str(reply)[:80]}")
    print(f"[CHAT] ─────────────────────────────────────────\n")

    # Sync complaint to memory (SQLite save happens inside graph.py)
    complaint_id = result.get("complaint_id")
    if complaint_id:
        print(f"[CHAT] complaint_id found: {complaint_id}")
        if complaint_id not in complaints_db:
            complaint = result.get("complaint_obj")
            if complaint:
                print(f"[CHAT] complaint_obj found — adding to memory + notifying dept")
                complaints_db[complaint_id] = complaint
                background_tasks.add_task(notify_department, complaint)
            else:
                print(f"[CHAT] complaint_obj is None — reloading from SQLite")
                loaded = load_complaint(complaint_id)
                if loaded:
                    print(f"[CHAT] ✓ Loaded {complaint_id} from SQLite")
                    complaints_db[complaint_id] = loaded
                    background_tasks.add_task(notify_department, loaded)
                else:
                    print(f"[CHAT] ✗ {complaint_id} NOT found in SQLite either!")
        else:
            print(f"[CHAT] {complaint_id} already in memory")
    else:
        print(f"[CHAT] No complaint_id in result — not a completed complaint")

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        complaint_id=complaint_id,
        state=result.get("graph_state"),
        suggestions=result.get("suggestions", []),
    )


# ── Status Tracking ─────────────────────────────────────────────────────────────

@app.get("/api/status/{complaint_id}")
async def get_status(complaint_id: str):
    complaint = complaints_db.get(complaint_id) or load_complaint(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return {
        "complaint_id": complaint_id,
        "status": complaint.status,
        "category": complaint.category,
        "priority": complaint.priority,
        "created_at": complaint.created_at,
        "updated_at": complaint.updated_at,
        "resolution_notes": complaint.resolution_notes,
        "estimated_resolution": complaint.estimated_resolution,
        "similar_issues_count": complaint.similar_count,
    }


# ── Officer Dashboard ───────────────────────────────────────────────────────────

@app.get("/api/officer/dashboard")
async def officer_dashboard():
    all_complaints = list(complaints_db.values())

    if not all_complaints:
        return OfficerDashboard(
            total=0, open=0, in_progress=0, resolved=0,
            critical=[], hot_zones=[], category_breakdown={}, recent=[]
        )

    priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1,
                      Priority.MEDIUM: 2, Priority.LOW: 3}
    sorted_complaints = sorted(
        all_complaints,
        key=lambda c: (priority_order.get(c.priority, 3), c.created_at)
    )

    zone_counts: dict[str, int] = {}
    for c in all_complaints:
        zone = c.zone or "Unknown"
        zone_counts[zone] = zone_counts.get(zone, 0) + 1

    hot_zones = [
        {"zone": z, "count": n}
        for z, n in sorted(zone_counts.items(), key=lambda x: -x[1])
        if n >= 2
    ]

    cat_breakdown: dict[str, int] = {}
    for c in all_complaints:
        cat = str(c.category.value) if c.category else "Unknown"
        cat_breakdown[cat] = cat_breakdown.get(cat, 0) + 1

    open_count = sum(1 for c in all_complaints if c.status == ComplaintStatus.OPEN)
    in_prog    = sum(1 for c in all_complaints if c.status == ComplaintStatus.IN_PROGRESS)
    resolved   = sum(1 for c in all_complaints if c.status == ComplaintStatus.RESOLVED)

    return OfficerDashboard(
        total=len(all_complaints),
        open=open_count,
        in_progress=in_prog,
        resolved=resolved,
        critical=[c for c in sorted_complaints if c.priority in (Priority.CRITICAL, Priority.HIGH)][:5],
        hot_zones=hot_zones,
        category_breakdown=cat_breakdown,
        recent=sorted(all_complaints, key=lambda c: c.created_at, reverse=True)[:10],
    )


@app.put("/api/officer/update/{complaint_id}")
async def update_complaint(complaint_id: str, update: dict):
    complaint = complaints_db.get(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    updated_at = datetime.now(timezone.utc).isoformat()
    db_fields = {"updated_at": updated_at}

    if "status" in update:
        complaint.status = ComplaintStatus(update["status"])
        db_fields["status"] = update["status"]
    if "resolution_notes" in update:
        complaint.resolution_notes = update["resolution_notes"]
        db_fields["resolution_notes"] = update["resolution_notes"]
    if "priority" in update:
        complaint.priority = Priority(update["priority"])
        db_fields["priority"] = update["priority"]
    if "assigned_to" in update:
        complaint.assigned_to = update["assigned_to"]
        db_fields["assigned_to"] = update["assigned_to"]
    if "similar_count" in update:
        complaint.similar_count = update["similar_count"]
        db_fields["similar_count"] = update["similar_count"]

    complaint.updated_at = updated_at
    update_complaint_field(complaint_id, **db_fields)
    return {"message": "Updated", "complaint": complaint}


@app.post("/api/officer/escalate")
async def escalate(req: EscalateRequest):
    complaint = complaints_db.get(req.complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    updated_at = datetime.now(timezone.utc).isoformat()
    complaint.status = ComplaintStatus.ESCALATED
    complaint.priority = Priority.CRITICAL
    complaint.resolution_notes = f"Escalated by {req.officer_id}: {req.reason}"
    complaint.updated_at = updated_at

    update_complaint_field(
        req.complaint_id,
        status=ComplaintStatus.ESCALATED.value,
        priority=Priority.CRITICAL.value,
        resolution_notes=complaint.resolution_notes,
        updated_at=updated_at,
    )
    return {"message": "Escalated", "complaint": complaint}


# ── Analytics ───────────────────────────────────────────────────────────────────

@app.get("/api/analytics/heatmap")
async def heatmap():
    points = []
    for c in complaints_db.values():
        if c.location and c.location.get("lat"):
            points.append({
                "lat": c.location["lat"],
                "lng": c.location["lng"],
                "category": str(c.category.value) if c.category else None,
                "priority": str(c.priority.value) if c.priority else None,
                "status": str(c.status.value) if c.status else None,
            })
    return {"points": points}


@app.get("/api/analytics/trends")
async def trends():
    daily: dict[str, dict] = {}
    for c in complaints_db.values():
        day = c.created_at[:10]
        if day not in daily:
            daily[day] = {"date": day, "total": 0, "resolved": 0}
        daily[day]["total"] += 1
        if c.status == ComplaintStatus.RESOLVED:
            daily[day]["resolved"] += 1
    return {"trends": sorted(daily.values(), key=lambda x: x["date"])}


# ── Helpers ─────────────────────────────────────────────────────────────────────

async def notify_department(complaint: Complaint):
    dept_map = {
        ComplaintCategory.POTHOLE:       "PWD",
        ComplaintCategory.GARBAGE:       "BBMP Sanitation",
        ComplaintCategory.WATER_LEAKAGE: "BWSSB",
        ComplaintCategory.ELECTRICITY:   "BESCOM",
        ComplaintCategory.STREETLIGHT:   "BESCOM",
        ComplaintCategory.SEWAGE:        "BWSSB",
        ComplaintCategory.TREE_FALL:     "Forest Department",
        ComplaintCategory.OTHER:         "General Services",
    }
    dept = dept_map.get(complaint.category, "General Services")
    print(f"[NOTIFY] Complaint {complaint.id} sent to {dept} — Priority: {complaint.priority}")


# ── Serve React Frontend ────────────────────────────────────────────────────────

DIST_DIR = str(Path(__file__).resolve().parent / "dist")
print(f"[STATIC] Looking for dist at: {DIST_DIR}")
print(f"[STATIC] dist exists: {os.path.exists(DIST_DIR)}")

if os.path.exists(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
else:
    print("[INFO] No dist/ folder found. Run `npm run build` in civic-frontend and copy dist/ here.")


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
