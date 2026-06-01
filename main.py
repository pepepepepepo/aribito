"""
Aribito — FastAPI backend
AI personas that simply exist.

Endpoints:
  GET  /health           — liveness check
  GET  /api/personas     — list all personas
  POST /api/chat         — chat with a persona (Ollama backend)
  POST /api/council      — all personas respond in parallel

Requirements: Ollama running locally with a model available.
Default model: gemma3:4b (change via OLLAMA_MODEL env var)
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List
import asyncio
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import PersonaInfo, ChatMessage, ChatResponse, CouncilMessage, CouncilResponse, CouncilVoice, CouncilHistoryRound
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


@app.post("/api/council", response_model=CouncilResponse)
async def council(req: CouncilMessage):
    """Send the same message to multiple personas in parallel via asyncio.gather."""
    all_personas = list_personas()

    if req.persona_ids:
        targets = [p for p in all_personas if p.id in req.persona_ids]
    else:
        targets = all_personas

    if not targets:
        raise HTTPException(status_code=404, detail="No matching personas found")

    async def ask_one(persona_info: PersonaInfo) -> CouncilVoice:
        pid = persona_info.id
        data = get_persona(pid)
        if data is None:
            return CouncilVoice(
                persona_id=pid,
                persona_name=persona_info.name,
                emoji=persona_info.emoji,
                role=persona_info.role,
                response="",
                error="Persona data not found",
            )
        system_prompt = build_system_prompt(data)
        messages = [{"role": "system", "content": system_prompt}]

        # Inject previous rounds: own responses as assistant, others as context note
        for round_ in (req.history or []):
            messages.append({"role": "user", "content": round_.message})
            my_voice = next(
                (v for v in round_.voices if v.persona_id == pid and v.response and not v.error),
                None,
            )
            if my_voice:
                messages.append({"role": "assistant", "content": my_voice.response})
            else:
                # Placeholder so the turn pair stays balanced
                messages.append({"role": "assistant", "content": "(no response)"})

        # Build current user message, appending last round's other-voices as context
        current_message = req.message
        if req.history:
            last_round = req.history[-1]
            others = [
                v for v in last_round.voices
                if v.persona_id != pid and v.response and not v.error
            ]
            if others:
                context_lines = "\n".join(
                    f"{v.emoji} {v.persona_name}: {v.response}" for v in others
                )
                current_message = (
                    f"{req.message}\n\n"
                    f"[Other council members said in the previous round:\n{context_lines}]"
                )

        messages.append({"role": "user", "content": current_message})
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
                )
                resp.raise_for_status()
                result = resp.json()
            reply = result.get("message", {}).get("content", "")
            return CouncilVoice(
                persona_id=pid,
                persona_name=data.get("name", ""),
                emoji=data.get("emoji", "🤖"),
                role=data.get("role", ""),
                response=reply,
            )
        except Exception as e:
            return CouncilVoice(
                persona_id=pid,
                persona_name=data.get("name", pid),
                emoji=data.get("emoji", "🤖"),
                role=data.get("role", ""),
                response="",
                error=str(e),
            )

    voices = await asyncio.gather(*[ask_one(p) for p in targets])
    model_used = OLLAMA_MODEL

    return CouncilResponse(
        message=req.message,
        voices=list(voices),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model_used,
    )


# Serve static files (chat UI) — MUST be mounted after all API routes
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
