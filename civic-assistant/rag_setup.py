"""
RAG Setup — Civic Knowledge Base
==================================
Uses LangChain + FAISS for:
  1. Retrieving relevant city policies/FAQs for context
  2. Semantic duplicate detection across complaints
  3. Resolution time estimates from historical data
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import json
import os


# ── Static Knowledge Base ──────────────────────────────────────────────────────

CIVIC_KNOWLEDGE = [
    # Policies
    """BBMP Complaint Resolution Policy:
    All complaints submitted through the civic portal are assigned a unique ID.
    Resolution SLAs: Critical issues (electrical hazards, fallen trees) — 4 hours.
    High priority (water leaks, sewage overflow) — 24 hours.
    Medium priority (potholes, garbage) — 72 hours.
    Low priority (minor streetlight issues) — 7 days.
    Citizens can escalate if not resolved within 150% of SLA time.""",

    """Pothole Reporting Guidelines:
    Report potholes using the civic portal with exact location (landmark, GPS coordinates preferred).
    For highway potholes, contact NHAI separately. City roads fall under BBMP PWD.
    Deep potholes (>6 inches) are classified HIGH priority. Potholes causing accidents = CRITICAL.
    Multiple potholes on same stretch can be reported as one complaint.""",

    """Garbage & Sanitation Issues:
    BBMP provides door-to-door garbage collection 6 days/week.
    Report missed collection, illegal dumping, or overflowing bins.
    Burning garbage is a serious violation — escalate to pollution control.
    Hospital/school area garbage = HIGH priority due to health hazard.""",

    """Water Supply & Leakage — BWSSB:
    Report pipe bursts, road flooding from leaks, no water supply.
    Burst mains affecting traffic = CRITICAL. Minor drips = MEDIUM.
    BWSSB helpline: 1916. Online portal: bwssb.gov.in
    Typical repair time: 24-48 hours for large mains, 72 hours for minor leaks.""",

    """Electricity Failures — BESCOM:
    Report power outages, sparking wires, fallen poles, transformer failures.
    Sparking/fire risks = CRITICAL — BESCOM responds within 2 hours.
    Planned outages are notified 24 hours in advance via SMS/app.
    BESCOM helpline: 1912.""",

    """Streetlight Issues:
    Report non-functional streetlights, especially on highways or dark alleys.
    Multiple lights out on one street = HIGH priority (safety concern).
    Single light out in well-lit area = LOW priority.
    Smart LED streetlights have auto-fault detection — report confirms field inspection.""",

    """Tree Fall & Horticulture:
    Fallen trees blocking roads = CRITICAL — Forest department responds in 2-4 hours.
    Report dangerous leaning trees before they fall.
    Post-monsoon tree inspection happens annually.
    Clear the area around fallen trees — do not attempt removal yourself.""",

    """Duplicate Complaint Policy:
    The system automatically detects similar complaints in the same zone.
    Duplicate reports increase the priority of the original complaint.
    Citizens are notified of the original complaint ID and current status.
    This helps officers identify problem hotspots faster.""",

    """Complaint Escalation:
    Citizens can escalate if SLA is breached.
    Escalated complaints go to senior officer / commissioner level.
    Escalation reason must be provided.
    Priority auto-upgraded to CRITICAL on escalation.""",

    """Geographic Hotspot Detection:
    The system tracks repeated complaints in the same zone.
    Areas with 5+ complaints of same type within 500m = declared hotspot.
    Hotspots get priority budget allocation in the next quarter.
    Officers receive weekly hotspot reports.""",
]


# ── RAG Class ─────────────────────────────────────────────────────────────────

class CivicRAG:
    """
    Manages two vector stores:
    1. `policy_store` — static civic knowledge (FAQs, policies, guidelines)
    2. `complaint_store` — dynamic complaint history for duplicate detection
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        print("[RAG] Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._build_policy_store()
        self._complaint_store: Optional[FAISS] = None
        self._complaint_meta: Dict[str, Any] = {}  # doc_id → complaint_id
        print("[RAG] Ready ✓")

    def _build_policy_store(self):
        """Ingest static civic knowledge into FAISS."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = []
        for i, text in enumerate(CIVIC_KNOWLEDGE):
            chunks = splitter.create_documents([text])
            for c in chunks:
                c.metadata["source"] = f"policy_{i}"
            docs.extend(chunks)

        self.policy_store = FAISS.from_documents(docs, self.embeddings)
        print(f"[RAG] Indexed {len(docs)} policy chunks")

    def get_context(self, query: str, k: int = 3) -> str:
        """Retrieve top-k relevant policy chunks for the given query."""
        try:
            results = self.policy_store.similarity_search(query, k=k)
            return "\n---\n".join([r.page_content for r in results])
        except Exception as e:
            return f"[RAG context unavailable: {e}]"

    def index_complaint(self, complaint) -> None:
        """Add a resolved/new complaint to the complaint vector store."""
        text = f"{complaint.category} in {complaint.zone}: {complaint.description}"
        doc = Document(
            page_content=text,
            metadata={
                "complaint_id": complaint.id,
                "category": str(complaint.category),
                "zone": complaint.zone or "",
                "status": str(complaint.status),
            }
        )

        if self._complaint_store is None:
            self._complaint_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self._complaint_store.add_documents([doc])

        self._complaint_meta[doc.page_content] = complaint.id
        print(f"[RAG] Indexed complaint {complaint.id}")

    def find_similar(
        self,
        description: str,
        candidates: list,
        threshold: float = 0.75,
    ) -> Optional[Dict[str, Any]]:
        """
        Semantic similarity check: returns the most similar complaint
        from candidates with score > threshold.
        """
        if not self._complaint_store or not candidates:
            return None

        try:
            results = self._complaint_store.similarity_search_with_score(description, k=5)
            for doc, score in results:
                # FAISS returns L2 distance — lower = more similar
                # Normalize to [0,1] similarity
                similarity = 1 / (1 + score)
                cid = doc.metadata.get("complaint_id")
                matched = next((c for c in candidates if c.id == cid), None)
                if matched and similarity >= threshold:
                    return {"complaint": matched, "score": similarity}
        except Exception as e:
            print(f"[RAG] Similarity search error: {e}")

        return None

    def get_resolution_estimate(self, category: str, zone: str) -> Optional[str]:
        """
        Query historical complaints to estimate resolution time.
        Returns a helpful estimate message.
        """
        query = f"resolution time for {category} in {zone}"
        context = self.get_context(query, k=2)
        return context

    def get_officer_briefing(self, zone: str) -> str:
        """Retrieve relevant policies/history for officer decision-making."""
        query = f"priority issues in {zone} municipal guidelines"
        return self.get_context(query, k=4)