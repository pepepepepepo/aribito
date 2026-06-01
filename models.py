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
