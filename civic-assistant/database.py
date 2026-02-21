"""
database.py — SQLite Persistence Layer
=======================================
Stores complaints and conversations in a local SQLite file (civic.db).
Data survives server restarts. No extra packages needed — sqlite3 is built into Python.
"""

import sqlite3
import json
import os
from typing import Optional
from datetime import datetime

from models import Complaint, ComplaintStatus, ComplaintCategory, Priority

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "civic.db")


# ── Connection ─────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


# ── Schema Setup ───────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = get_conn()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS complaints (
            id                  TEXT PRIMARY KEY,
            session_id          TEXT,
            description         TEXT,
            category            TEXT,
            priority            TEXT,
            status              TEXT,
            location            TEXT,   -- JSON string
            zone                TEXT,
            created_at          TEXT,
            updated_at          TEXT,
            resolved_at         TEXT,
            resolution_notes    TEXT,
            estimated_resolution TEXT,
            assigned_to         TEXT,
            similar_count       INTEGER DEFAULT 0,
            duplicate_of        TEXT,
            citizen_contact     TEXT,
            images              TEXT    -- JSON array string
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,   -- 'user' or 'assistant'
            content     TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_complaints_status   ON complaints(status);
        CREATE INDEX IF NOT EXISTS idx_complaints_zone     ON complaints(zone);
        CREATE INDEX IF NOT EXISTS idx_complaints_category ON complaints(category);
        CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
    """)

    conn.commit()
    conn.close()
    print(f"[DB] SQLite initialized at: {DB_PATH}")


# ── Complaint Operations ───────────────────────────────────────────────────────

def save_complaint(complaint: Complaint):
    """Insert or update a complaint in the database."""
    print(f"[DB] Saving complaint {complaint.id} to {DB_PATH}")
    conn = get_conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO complaints
                (id, session_id, description, category, priority, status,
                 location, zone, created_at, updated_at, resolved_at,
                 resolution_notes, estimated_resolution, assigned_to,
                 similar_count, duplicate_of, citizen_contact, images)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            complaint.id,
            complaint.session_id,
            complaint.description,
            str(complaint.category.value) if complaint.category else None,
            str(complaint.priority.value) if complaint.priority else None,
            str(complaint.status.value) if complaint.status else None,
            json.dumps(complaint.location) if complaint.location else None,
            complaint.zone,
            complaint.created_at,
            complaint.updated_at,
            complaint.resolved_at,
            complaint.resolution_notes,
            complaint.estimated_resolution,
            complaint.assigned_to,
            complaint.similar_count,
            complaint.duplicate_of,
            complaint.citizen_contact,
            json.dumps(complaint.images) if complaint.images else "[]",
        ))
        conn.commit()
        print(f"[DB] ✓ Committed complaint {complaint.id}")
    except Exception as e:
        conn.rollback()
        print(f"[DB] ✗ Error saving complaint {complaint.id}: {e}")
        raise
    finally:
        conn.close()


def load_complaint(complaint_id: str) -> Optional[Complaint]:
    """Load a single complaint by ID."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM complaints WHERE id = ?", (complaint_id,)
    ).fetchone()
    conn.close()
    return _row_to_complaint(row) if row else None


def load_all_complaints() -> dict[str, Complaint]:
    """Load all complaints into a dict keyed by ID (mirrors in-memory structure)."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM complaints").fetchall()
    conn.close()
    return {row["id"]: _row_to_complaint(row) for row in rows}


def update_complaint_field(complaint_id: str, **fields):
    """Update specific fields on a complaint."""
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [complaint_id]
    conn = get_conn()
    conn.execute(
        f"UPDATE complaints SET {set_clause} WHERE id = ?", values
    )
    conn.commit()
    conn.close()


def increment_similar_count(complaint_id: str):
    """Atomically increment the similar_count for duplicate tracking."""
    conn = get_conn()
    conn.execute(
        "UPDATE complaints SET similar_count = similar_count + 1 WHERE id = ?",
        (complaint_id,)
    )
    conn.commit()
    conn.close()


def _row_to_complaint(row: sqlite3.Row) -> Complaint:
    """Convert a SQLite row to a Complaint model."""
    return Complaint(
        id=row["id"],
        session_id=row["session_id"] or "",
        description=row["description"] or "",
        category=ComplaintCategory(row["category"]) if row["category"] else None,
        priority=Priority(row["priority"]) if row["priority"] else Priority.MEDIUM,
        status=ComplaintStatus(row["status"]) if row["status"] else ComplaintStatus.OPEN,
        location=json.loads(row["location"]) if row["location"] else None,
        zone=row["zone"] or "",
        created_at=row["created_at"] or datetime.utcnow().isoformat(),
        updated_at=row["updated_at"] or datetime.utcnow().isoformat(),
        resolved_at=row["resolved_at"],
        resolution_notes=row["resolution_notes"],
        estimated_resolution=row["estimated_resolution"],
        assigned_to=row["assigned_to"],
        similar_count=row["similar_count"] or 0,
        duplicate_of=row["duplicate_of"],
        citizen_contact=row["citizen_contact"],
        images=json.loads(row["images"]) if row["images"] else [],
    )


# ── Conversation Operations ────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str):
    """Append a single message to the conversation history."""
    conn = get_conn()
    conn.execute("""
        INSERT INTO conversations (session_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def load_conversation(session_id: str) -> list[dict]:
    """Load full conversation history for a session."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT role, content FROM conversations
        WHERE session_id = ?
        ORDER BY id ASC
    """, (session_id,)).fetchall()
    conn.close()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def get_all_sessions() -> list[str]:
    """List all unique session IDs."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT session_id FROM conversations"
    ).fetchall()
    conn.close()
    return [row["session_id"] for row in rows]


def test_db():
    """Quick sanity check — run with: python database.py"""
    print(f"[TEST] DB path: {DB_PATH}")
    init_db()
    all_complaints = load_all_complaints()
    print(f"[TEST] Complaints in DB: {len(all_complaints)}")
    for c in all_complaints.values():
        print(f"  → {c.id} | {c.category} | {c.status} | {c.zone}")


if __name__ == "__main__":
    test_db()
