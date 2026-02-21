"""
LangGraph Complaint Workflow
============================

State machine flow:
  INTAKE → CATEGORIZE → DUPLICATE_CHECK → PRIORITY_ASSIGN → CONFIRM → DONE
                                ↓
                          DUPLICATE_FOUND (link & notify)

Each node is an async function that transforms ComplaintState.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal, TypedDict, Annotated
from datetime import datetime
import uuid
import json
import re

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from models import (
    Complaint, ComplaintCategory, Priority, ComplaintStatus,
    compute_priority, sla_message, PRIORITY_RULES
)
from rag_setup import CivicRAG

# Import DB saver — only used when SQLite is enabled
try:
    from database import save_complaint as db_save_complaint
    DB_ENABLED = True
except ImportError:
    DB_ENABLED = False

import os

# ── LLM Provider Config ────────────────────────────────────────────────────────
#
#  Set LLM_PROVIDER in your .env file:
#    LLM_PROVIDER=ollama  →  Llama 3.2:3b locally via Ollama (free, offline)
#    LLM_PROVIDER=groq    →  Llama 3.3 70b via Groq cloud (free tier, very fast)
#
#  For Groq: GROQ_API_KEY=gsk_...   →  https://console.groq.com
#
LLM_PROVIDER   = os.getenv("LLM_PROVIDER",   "ollama").lower()
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "llama3.2:3b")
OLLAMA_BASE_URL= os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
GROQ_MODEL     = os.getenv("GROQ_MODEL",     "llama-3.3-70b-versatile")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")


# ── State Schema ───────────────────────────────────────────────────────────────

class ComplaintState(TypedDict):
    # Conversation
    session_id: str
    messages: Annotated[list, add_messages]
    location: Optional[Dict[str, Any]]

    # Extracted complaint data
    raw_description: Optional[str]
    category: Optional[str]
    priority: Optional[str]
    zone: Optional[str]

    # Graph control
    graph_state: str   # current node name
    intent: Optional[str]  # "report" | "track" | "info" | "escalate"
    complaint_id: Optional[str]
    complaint_obj: Optional[Complaint]
    duplicate_of: Optional[str]

    # Output
    reply: str
    suggestions: List[str]

    # Injected deps
    complaints_db: Dict[str, Complaint]
    rag: Any


# ── LLM Factory ────────────────────────────────────────────────────────────────

def get_llm():
    """
    Returns the configured LLM based on LLM_PROVIDER env var.

    Ollama  → local Llama 3.2:3b  (free, offline, no API key needed)
    Groq    → cloud Llama 3.3 70b (free tier, ~10x faster than local 3b)
    """
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "Get your free key at https://console.groq.com\n"
                "Then add to .env:  GROQ_API_KEY=gsk_..."
            )
        print(f"[LLM] Using Groq → {GROQ_MODEL}")
        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0.3,
            api_key=GROQ_API_KEY,
        )
    else:
        print(f"[LLM] Using Ollama → {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.3,
            base_url=OLLAMA_BASE_URL,
        )


# ── Node: Intake ───────────────────────────────────────────────────────────────

async def intake_node(state: ComplaintState) -> ComplaintState:
    """
    Greet user, detect intent (report / track / info / exit), and extract initial details.
    """
    llm = get_llm()
    history = state["messages"]
    last_msg = (history[-1]["content"] if history else "").lower()

    # Fast exit detection — no LLM call needed
    exit_phrases = [
        "thank you", "thanks", "bye", "goodbye", "that's it", "thats it",
        "done", "ok thanks", "okay thanks", "no thanks", "that will be all",
        "nothing else", "i'm good", "im good", "all good", "great thanks",
    ]
    if any(phrase in last_msg for phrase in exit_phrases):
        return {
            **state,
            "graph_state": "done",
            "intent": "exit",
            "reply": "You're welcome! 😊 Have a great day. Your complaint is registered and you'll be notified of updates. Feel free to come back anytime to report issues or track status.",
            "suggestions": ["Report another issue", "Track my complaint"],
        }

    system = """You are CivicBot, a friendly assistant for the Bengaluru Municipal Corporation.
Your job is to help citizens report civic issues (potholes, garbage, water leaks, electricity failures, streetlights, sewage, tree falls).

Detect intent from the user's message:
- "report": They want to report a new issue
- "track": They want to track an existing complaint (they'll have a complaint ID)
- "info": They want general information
- "escalate": They want to escalate an existing complaint
- "exit": They are saying goodbye, thank you, or ending the conversation

Respond in JSON:
{
  "intent": "report|track|info|escalate|exit",
  "reply": "<friendly response in 2-3 sentences, ask for more details if needed>",
  "needs_location": true/false,
  "needs_description": true/false
}

If intent is "exit": reply with a warm goodbye, do NOT ask for more details.
If they're reporting: ask for the location and description if not provided.
Keep responses concise and empathetic."""

    # RAG context for greeting
    rag_context = state["rag"].get_context("civic issue reporting help")

    resp = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=f"User message: {last_msg}\n\nContext: {rag_context}")
    ])

    try:
        data = json.loads(resp.content)
    except Exception:
        # Fallback
        data = {
            "intent": "report",
            "reply": "Hello! I'm CivicBot 🏙️. I'm here to help you report civic issues in your area. Could you please describe the problem and let me know the location?",
            "needs_location": True,
            "needs_description": True,
        }

    suggestions = []
    if data.get("intent") == "report":
        suggestions = ["🕳️ Pothole", "🗑️ Garbage", "💧 Water Leak", "⚡ Electricity", "🔦 Streetlight", "🌊 Sewage", "🌳 Tree Fall"]

    return {
        **state,
        "graph_state": "intake",
        "intent": data["intent"],
        "reply": data["reply"],
        "suggestions": suggestions,
    }


# ── Node: Categorize ───────────────────────────────────────────────────────────

async def categorize_node(state: ComplaintState) -> ComplaintState:
    """
    Extract structured complaint data: category, location details, severity.
    Uses RAG to enrich context with past similar issues.
    """
    llm = get_llm()
    history = state["messages"]
    conversation = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-6:]])

    # Build RAG context from the description
    raw_desc = state.get("raw_description") or history[-1]["content"]
    rag_context = state["rag"].get_context(raw_desc)

    system = f"""You are an AI that categorizes civic complaints for Bengaluru Municipal Corporation.

Categories: POTHOLE, GARBAGE, WATER_LEAKAGE, ELECTRICITY, STREETLIGHT, SEWAGE, TREE_FALL, ROAD_DAMAGE, NOISE, OTHER

Based on the conversation, extract:
1. Category (from list above)
2. Precise description of the issue
3. Location/zone details  
4. Any severity indicators

Similar past complaints for context:
{rag_context}

Respond ONLY in JSON:
{{
  "category": "CATEGORY",
  "description": "clean description",
  "location_desc": "street/area name",
  "zone": "area zone (e.g., Koramangala, Whitefield)",
  "severity_keywords": ["list", "of", "keywords"],
  "confidence": 0.0-1.0,
  "reply": "Confirm what you understood + ask anything missing",
  "is_complete": true/false
}}"""

    resp = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Conversation:\n{conversation}")
    ])

    try:
        data = json.loads(resp.content)
    except Exception:
        data = {
            "category": "OTHER",
            "description": raw_desc,
            "location_desc": "",
            "zone": "",
            "severity_keywords": [],
            "confidence": 0.5,
            "reply": "I've noted your complaint. Could you confirm the exact location?",
            "is_complete": False,
        }

    category_str = data.get("category", "OTHER")
    try:
        category = ComplaintCategory(category_str)
    except ValueError:
        category = ComplaintCategory.OTHER

    desc = data.get("description", raw_desc)
    priority = compute_priority(category, desc)

    return {
        **state,
        "graph_state": "categorize",
        "category": category_str,
        "raw_description": desc,
        "zone": data.get("zone") or (state["location"] or {}).get("address", ""),
        "reply": data["reply"],
        "priority": priority.value,
        "suggestions": [],
    }


# ── Node: Duplicate Check ──────────────────────────────────────────────────────

async def duplicate_check_node(state: ComplaintState) -> ComplaintState:
    """
    Detect if an identical or very similar complaint already exists.
    Uses semantic similarity via RAG + zone + category matching.
    """
    complaints_db = state["complaints_db"]
    rag = state["rag"]

    category = state.get("category", "OTHER")
    zone = state.get("zone", "")
    description = state.get("raw_description", "")

    similar_complaints = []
    for cid, c in complaints_db.items():
        if (
            c.category == category
            and c.status not in (ComplaintStatus.RESOLVED, ComplaintStatus.CLOSED)
            and (not zone or c.zone == zone)
        ):
            similar_complaints.append(c)

    if similar_complaints:
        # Use RAG similarity to check semantic overlap
        top_similar = rag.find_similar(description, similar_complaints)

        if top_similar and top_similar["score"] > 0.8:
            # Clear duplicate
            dup = top_similar["complaint"]
            dup.similar_count += 1
            return {
                **state,
                "graph_state": "duplicate_found",
                "duplicate_of": dup.id,
                "reply": (
                    f"🔍 We found an existing complaint (#{dup.id}) for a similar issue in {zone or 'your area'}. "
                    f"Your concern has been linked to it — this increases its priority! "
                    f"Current status: **{dup.status}**. "
                    f"Track it anytime with ID: **{dup.id}**"
                ),
                "suggestions": [f"Track #{dup.id}", "Report different issue"],
            }

        # Similar but not duplicate — note them
        return {
            **state,
            "graph_state": "categorize",
            "reply": (
                f"✅ Noted. There are {len(similar_complaints)} related report(s) nearby — "
                "yours is being registered as a new complaint to capture the full extent of the problem."
            ),
            "suggestions": [],
        }

    return {
        **state,
        "graph_state": "categorize",
        "reply": None,   # continue to priority assignment
        "suggestions": [],
    }


# ── Node: Priority & Assignment ────────────────────────────────────────────────

async def priority_assign_node(state: ComplaintState) -> ComplaintState:
    """
    Finalize priority, create Complaint object, assign to department.
    """
    category_str = state.get("category", "OTHER")
    priority_str = state.get("priority", Priority.MEDIUM.value)

    try:
        category = ComplaintCategory(category_str)
        priority = Priority(priority_str)
    except ValueError:
        category = ComplaintCategory.OTHER
        priority = Priority.MEDIUM

    # Department routing
    dept_map = {
        ComplaintCategory.POTHOLE: "Public Works Department",
        ComplaintCategory.GARBAGE: "BBMP Sanitation",
        ComplaintCategory.WATER_LEAKAGE: "BWSSB Water Supply",
        ComplaintCategory.ELECTRICITY: "BESCOM Electrical",
        ComplaintCategory.STREETLIGHT: "BESCOM Street Lighting",
        ComplaintCategory.SEWAGE: "BWSSB Drainage",
        ComplaintCategory.TREE_FALL: "Forest & Horticulture Dept",
        ComplaintCategory.ROAD_DAMAGE: "Public Works Department",
        ComplaintCategory.NOISE: "Pollution Control Board",
        ComplaintCategory.OTHER: "General Municipal Services",
    }
    dept = dept_map.get(category, "General Municipal Services")
    sla = sla_message(priority)

    complaint = Complaint(
        session_id=state["session_id"],
        description=state.get("raw_description", ""),
        category=category,
        priority=priority,
        status=ComplaintStatus.OPEN,
        location=state.get("location"),
        zone=state.get("zone", ""),
        estimated_resolution=sla,
        assigned_to=dept,
    )

    # Add to RAG index for future duplicate detection
    state["rag"].index_complaint(complaint)

    # Save to SQLite immediately if DB is enabled
    if DB_ENABLED:
        try:
            db_save_complaint(complaint)
            print(f"[DB] ✓ Saved complaint {complaint.id} to SQLite")
        except Exception as e:
            import traceback
            print(f"[DB] ✗ Failed to save complaint {complaint.id}: {e}")
            traceback.print_exc()

    priority_emoji = {
        Priority.CRITICAL: "🔴",
        Priority.HIGH: "🟠",
        Priority.MEDIUM: "🟡",
        Priority.LOW: "🟢",
    }

    reply = (
        f"✅ **Complaint Registered Successfully!**\n\n"
        f"📋 **ID:** `{complaint.id}`\n"
        f"🏷️ **Category:** {category_str.replace('_', ' ').title()}\n"
        f"{priority_emoji[priority]} **Priority:** {priority_str}\n"
        f"🏢 **Assigned to:** {dept}\n"
        f"⏱️ **Expected resolution:** {sla}\n\n"
        f"You'll receive updates as the status changes. Save your complaint ID: **{complaint.id}**"
    )

    return {
        **state,
        "graph_state": "done",
        "complaint_id": complaint.id,
        "complaint_obj": complaint,
        "reply": reply,
        "suggestions": [f"Track #{complaint.id}", "Report another issue", "See nearby issues"],
    }


# ── Routing Logic ──────────────────────────────────────────────────────────────

def route_intent(state: ComplaintState) -> str:
    intent = state.get("intent", "report")
    graph_state = state.get("graph_state", "intake")

    if intent in ("track", "info", "escalate", "exit"):
        return END
    if graph_state == "duplicate_found":
        return END
    if graph_state == "done":
        return END
    return "categorize"


def route_after_categorize(state: ComplaintState) -> str:
    if state.get("duplicate_of"):
        return END
    if state.get("graph_state") == "done":
        return END
    return "duplicate_check"


def route_after_duplicate(state: ComplaintState) -> str:
    if state.get("duplicate_of"):
        return END
    return "priority_assign"


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_complaint_graph(rag: CivicRAG):
    """
    Build and compile the LangGraph state machine.

    Flow:
        START → intake → [route by intent]
                       → categorize → duplicate_check → priority_assign → END
                       → intake (for track/info)
    """
    builder = StateGraph(ComplaintState)

    # Register nodes
    builder.add_node("intake", intake_node)
    builder.add_node("categorize", categorize_node)
    builder.add_node("duplicate_check", duplicate_check_node)
    builder.add_node("priority_assign", priority_assign_node)

    # Define edges
    builder.add_edge(START, "intake")

    builder.add_conditional_edges(
        "intake",
        route_intent,
        {
            "categorize": "categorize",
            "intake": END,
            "duplicate_check": "duplicate_check",
        }
    )

    builder.add_conditional_edges(
        "categorize",
        route_after_categorize,
        {
            "duplicate_check": "duplicate_check",
            "categorize": END,
            END: END,
        }
    )

    builder.add_conditional_edges(
        "duplicate_check",
        route_after_duplicate,
        {
            "priority_assign": "priority_assign",
            END: END,
        }
    )

    builder.add_edge("priority_assign", END)

    return builder.compile()
