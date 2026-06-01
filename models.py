"""
Aribito — Pydantic models
Lightweight. No goton, no resonance, no fracture.
"""
from pydantic import BaseModel
from typing import List, Optional


class PersonaInfo(BaseModel):
    id: str
    name: str
    emoji: Optional[str] = "🤖"
    role: str
    description: str
    color: Optional[str] = "#7B68EE"


class ChatMessage(BaseModel):
    persona_id: str
    message: str
    conversation_history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    persona_name: str
    emoji: Optional[str] = "🤖"
    response: str
    timestamp: str
    model: Optional[str] = None


# ── Council (parallel multi-persona) ──

class CouncilVoice(BaseModel):
    persona_id: str
    persona_name: str
    emoji: Optional[str] = "🤖"
    role: str
    response: str
    error: Optional[str] = None


class CouncilHistoryRound(BaseModel):
    """One round of council: the user message + all voices that responded."""
    message: str
    voices: List[CouncilVoice]


class CouncilMessage(BaseModel):
    message: str
    persona_ids: Optional[List[str]] = None  # None = all personas
    history: Optional[List[CouncilHistoryRound]] = []  # previous rounds


class CouncilResponse(BaseModel):
    message: str
    voices: List[CouncilVoice]
    timestamp: str
    model: Optional[str] = None
