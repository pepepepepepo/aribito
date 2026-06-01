"""
Aribito — FastAPI backend
AI personas that simply exist.

Endpoints:
  GET  /health           — liveness check
  GET  /api/personas     — list all personas
  POST /api/chat         — chat with a persona (Ollama backend)

Requirements: Ollama running locally with a model available.
Default model: gemma3:4b (change via OLLAMA_MODEL env var)
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import PersonaInfo, ChatMessage, ChatResponse
from core.persona_loader import list_personas, get_persona, build_system_prompt

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")


@asynccontextmanager
async def lifespan(app: FastAPI):
    personas = list_personas()
    print(f"[startup] {len(personas)} personas loaded", flush=True)
    yield


app = FastAPI(
    title="Aribito",
    description="AI personas that simply exist.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/personas", response_model=List[PersonaInfo])
async def get_personas():
    return list_personas()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage):
    data = get_persona(msg.persona_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Persona '{msg.persona_id}' not found")

    system_prompt = build_system_prompt(data)

    # Build messages for Ollama
    messages = [{"role": "system", "content": system_prompt}]
    for h in (msg.conversation_history or []):
        role = h.get("role", "user")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": msg.message})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            result = resp.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama at {OLLAMA_URL}. Is it running?",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    reply = result.get("message", {}).get("content", "")
    model_used = result.get("model", OLLAMA_MODEL)

    return ChatResponse(
        persona_name=data.get("name", ""),
        emoji=data.get("emoji", "🤖"),
        response=reply,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model_used,
    )
