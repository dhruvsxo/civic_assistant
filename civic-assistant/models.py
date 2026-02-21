"""
Models — Pydantic schemas for complaints, status, priority, and dashboard.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class ComplaintStatus(str, Enum):
    OPEN        = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    ESCALATED   = "ESCALATED"
    RESOLVED    = "RESOLVED"
    DUPLICATE   = "DUPLICATE"
    CLOSED      = "CLOSED"


class ComplaintCategory(str, Enum):
    POTHOLE      = "POTHOLE"
    GARBAGE      = "GARBAGE"
    WATER_LEAKAGE= "WATER_LEAKAGE"
    ELECTRICITY  = "ELECTRICITY"
    STREETLIGHT  = "STREETLIGHT"
    SEWAGE       = "SEWAGE"
    TREE_FALL    = "TREE_FALL"
    ROAD_DAMAGE  = "ROAD_DAMAGE"
    NOISE        = "NOISE"
    OTHER        = "OTHER"


class Priority(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class Complaint(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    session_id: str = ""
    description: str = ""
    category: Optional[ComplaintCategory] = None
    priority: Priority = Priority.MEDIUM
    status: ComplaintStatus = ComplaintStatus.OPEN
    location: Optional[Dict[str, Any]] = None
    zone: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    resolved_at: Optional[str] = None
    resolution_notes: Optional[str] = None
    estimated_resolution: Optional[str] = None
    assigned_to: Optional[str] = None
    similar_count: int = 0
    duplicate_of: Optional[str] = None
    citizen_contact: Optional[str] = None
    images: List[str] = []


class OfficerDashboard(BaseModel):
    total: int
    open: int
    in_progress: int
    resolved: int
    critical: List[Complaint]
    hot_zones: List[Dict[str, Any]]
    category_breakdown: Dict[str, int]
    recent: List[Complaint]


# Priority rules — auto-assigned based on category + severity keywords
PRIORITY_RULES = {
    ComplaintCategory.ELECTRICITY: {
        "default": Priority.HIGH,
        "keywords": {
            "fire": Priority.CRITICAL,
            "spark": Priority.CRITICAL,
            "shock": Priority.CRITICAL,
            "transformer": Priority.HIGH,
        }
    },
    ComplaintCategory.WATER_LEAKAGE: {
        "default": Priority.HIGH,
        "keywords": {
            "flood": Priority.CRITICAL,
            "burst": Priority.CRITICAL,
            "road": Priority.HIGH,
        }
    },
    ComplaintCategory.SEWAGE: {
        "default": Priority.HIGH,
        "keywords": {
            "overflow": Priority.CRITICAL,
            "block": Priority.HIGH,
        }
    },
    ComplaintCategory.TREE_FALL: {
        "default": Priority.CRITICAL,
        "keywords": {}
    },
    ComplaintCategory.POTHOLE: {
        "default": Priority.MEDIUM,
        "keywords": {
            "deep": Priority.HIGH,
            "accident": Priority.CRITICAL,
            "highway": Priority.HIGH,
        }
    },
    ComplaintCategory.GARBAGE: {
        "default": Priority.LOW,
        "keywords": {
            "hospital": Priority.HIGH,
            "school": Priority.HIGH,
            "stench": Priority.MEDIUM,
            "burning": Priority.CRITICAL,
        }
    },
    ComplaintCategory.STREETLIGHT: {
        "default": Priority.MEDIUM,
        "keywords": {
            "dark": Priority.HIGH,
            "highway": Priority.HIGH,
            "all": Priority.HIGH,
        }
    },
}


def compute_priority(category: ComplaintCategory, description: str) -> Priority:
    """Auto-compute priority from category + description keywords."""
    rules = PRIORITY_RULES.get(category)
    if not rules:
        return Priority.MEDIUM

    desc_lower = description.lower()
    for kw, p in rules["keywords"].items():
        if kw in desc_lower:
            return p
    return rules["default"]


# SLA (hours) by priority
SLA_HOURS = {
    Priority.CRITICAL: 4,
    Priority.HIGH: 24,
    Priority.MEDIUM: 72,
    Priority.LOW: 168,
}

# Estimated resolution message
def sla_message(priority: Priority) -> str:
    hours = SLA_HOURS[priority]
    if hours < 24:
        return f"within {hours} hours"
    days = hours // 24
    return f"within {days} day{'s' if days > 1 else ''}"
