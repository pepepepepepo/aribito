"""
Aribito — FastAPI backend
AI personas that simply exist.

Endpoints:
  GET  /health           — liveness check
  GET  /api/personas     — list all personas
  POST /api/chat         — chat with a persona (Ollama backend)
  POST /api/council      — all personas respond in parallel

Council uses a two-phase architecture:
  Phase 1 (Think): 3x THINK_MODEL in parallel → structured thought packets (non-verbal)
  Phase 2 (Voice): 3x VOICE_MODEL in parallel → each persona speaks using own + others' thought
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List
import asyncio
import json
import os
import re

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import PersonaInfo, ChatMessage, ChatResponse, CouncilMessage, CouncilResponse, CouncilVoice, CouncilHistoryRound
from core.persona_loader import list_personas, get_persona, build_system_prompt

OLLAMA_URL  = os.environ.get("OLLAMA_URL",  "http://localhost:11434")
THINK_MODEL = os.environ.get("THINK_MODEL", "qwen3:1.7b")    # lightweight — internal reasoning
VOICE_MODEL = os.environ.get("VOICE_MODEL", "qwen3.5:9b")   # quality     — articulated speech
# Legacy single-model fallback for /api/chat
CHAT_MODEL  = os.environ.get("CHAT_MODEL",  VOICE_MODEL)


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
                json={"model": CHAT_MODEL, "messages": messages, "stream": False},
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
    model_used = result.get("model", CHAT_MODEL)

    return ChatResponse(
        persona_name=data.get("name", ""),
        emoji=data.get("emoji", "🤖"),
        response=reply,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model_used,
    )


@app.post("/api/council", response_model=CouncilResponse)
async def council(req: CouncilMessage):
    """Two-phase council: Think (e2b, non-verbal) → Voice (e4b, articulated)."""
    all_personas = list_personas()

    if req.persona_ids:
        targets = [p for p in all_personas if p.id in req.persona_ids]
    else:
        targets = all_personas

    if not targets:
        raise HTTPException(status_code=404, detail="No matching personas found")

    # ── Phase 1: Think ────────────────────────────────────────────────────────
    # Each persona reasons internally using THINK_MODEL → structured JSON packet

    async def think_one(persona_info: PersonaInfo) -> dict:
        pid = persona_info.id
        data = get_persona(pid)
        if data is None:
            return {"persona_id": pid, "persona_name": persona_info.name,
                    "emoji": persona_info.emoji, "error": "Persona data not found"}

        system_prompt = build_system_prompt(data)
        per_think_model = data.get("think_model", THINK_MODEL)
        think_instruction = (
            "You are in your internal thinking phase. Do NOT write a response yet.\n"
            "Output ONLY a JSON object (no prose, no markdown) with:\n"
            '{"stance": "agree|disagree|neutral|questioning|moved|uncertain",\n'
            ' "concepts": ["2-4 key ideas you are focused on"],\n'
            ' "emotion": {"<label>": <0.0-1.0>, ...},\n  // 2-3 dimensions\n'
            ' "core_insight": "one sentence: what you most want to express"}'
        )

        messages = [{"role": "system", "content": system_prompt + "\n\n" + think_instruction}]
        for round_ in (req.history or []):
            messages.append({"role": "user", "content": round_.message})
            my_voice = next(
                (v for v in round_.voices if v.persona_id == pid and v.response and not v.error),
                None,
            )
            messages.append({"role": "assistant",
                             "content": my_voice.response if my_voice else "(no response)"})
        messages.append({"role": "user", "content": req.message})

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": per_think_model, "messages": messages, "stream": False},
                )
                resp.raise_for_status()
                raw = resp.json().get("message", {}).get("content", "{}")
            # Strip <think>...</think> tags (qwen3 reasoning traces)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            packet = json.loads(match.group()) if match else {}
        except Exception as e:
            packet = {"stance": "neutral", "concepts": [], "emotion": {}, "core_insight": str(e)}

        packet["persona_id"]   = pid
        packet["persona_name"] = data.get("name", pid)
        packet["emoji"]        = data.get("emoji", "🤖")
        return packet

    thought_results = await asyncio.gather(*[think_one(p) for p in targets],
                                           return_exceptions=True)
    thoughts: list[dict] = [
        t if isinstance(t, dict) else {"persona_id": targets[i].id, "error": str(t)}
        for i, t in enumerate(thought_results)
    ]

    # ── Phase 2: Voice ────────────────────────────────────────────────────────
    # Each persona articulates using VOICE_MODEL, informed by own + others' thought packets

    async def voice_one(persona_info: PersonaInfo, own: dict, others: list[dict]) -> CouncilVoice:
        pid = persona_info.id
        data = get_persona(pid)
        if data is None:
            return CouncilVoice(persona_id=pid, persona_name=persona_info.name,
                                emoji=persona_info.emoji, role=persona_info.role,
                                response="", error="Persona data not found")

        system_prompt = build_system_prompt(data)

        # Non-verbal signals from other council members
        others_signal = "\n".join(
            f"- {o.get('emoji','🤖')} {o.get('persona_name','?')}: "
            f"stance={o.get('stance','?')}, "
            f"emotion={o.get('emotion',{})}, "
            f"core_insight=\"{o.get('core_insight','')}\""
            for o in others if not o.get("error")
        )
        voice_context = (
            f"[Your internal thought]\n"
            f"  stance: {own.get('stance','?')}\n"
            f"  core_insight: {own.get('core_insight','')}\n"
            f"  emotion: {own.get('emotion',{})}\n"
            f"  concepts: {', '.join(own.get('concepts', []))}\n"
        )
        if others_signal:
            voice_context += f"\n[Non-verbal signals from the other council members]\n{others_signal}\n"
        voice_context += "\nNow speak — in your own voice and character."

        messages = [{"role": "system", "content": system_prompt}]
        for round_ in (req.history or []):
            messages.append({"role": "user", "content": round_.message})
            my_voice = next(
                (v for v in round_.voices if v.persona_id == pid and v.response and not v.error),
                None,
            )
            messages.append({"role": "assistant",
                             "content": my_voice.response if my_voice else "(no response)"})
        messages.append({"role": "user", "content": req.message + "\n\n" + voice_context})

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": VOICE_MODEL,
                        "messages": messages,
                        "stream": False,
                        "think": False,  # disable thinking mode (qwen3 family)
                        "options": {
                            "stop": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
                        },
                    },
                )
                resp.raise_for_status()
                reply = resp.json().get("message", {}).get("content", "")
                # Strip <think>...</think> tags (qwen3/qwen3.5 reasoning traces)
                reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
                # Strip special tokens that leak from some models (e.g. <|endoftext|><|im_start|>user …)
                reply = re.sub(r"<\|[a-z_]+\|>.*", "", reply, flags=re.DOTALL).strip()
                # Strip *meta-annotation* lines (model narrating its own instructions)
                reply = re.sub(r"\*[^*]{0,80}\*\s*$", "", reply, flags=re.MULTILINE).strip()
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
                response="", error=str(e),
            )

    voice_tasks = [
        voice_one(
            targets[i],
            thoughts[i],
            [t for j, t in enumerate(thoughts) if j != i],
        )
        for i in range(len(targets))
    ]
    voices = await asyncio.gather(*voice_tasks)

    return CouncilResponse(
        message=req.message,
        voices=list(voices),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=f"think:{THINK_MODEL} / voice:{VOICE_MODEL}",
    )


# Serve static files (chat UI) — MUST be mounted after all API routes
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
