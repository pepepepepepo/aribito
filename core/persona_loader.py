"""
Aribito — Persona Loader
Reads YAML files from personas/ directory.
No cache complexity, no goton, no resonance.
"""
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from models import PersonaInfo

PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"

_cache: Optional[Dict[str, dict]] = None


def _load_all() -> Dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache

    result: Dict[str, dict] = {}
    for yaml_file in sorted(PERSONAS_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            pid = str(data.get("id", yaml_file.stem))
            result[pid] = data
        except Exception as e:
            print(f"[warn] failed to load {yaml_file.name}: {e}", flush=True)

    _cache = result
    return _cache


def list_personas() -> List[PersonaInfo]:
    personas = _load_all()
    items = []
    for data in personas.values():
        items.append(PersonaInfo(
            id=str(data.get("id", "")),
            name=data.get("name", ""),
            emoji=data.get("emoji", "🤖"),
            role=data.get("role", ""),
            description=data.get("description", ""),
            color=data.get("color", "#7B68EE"),
        ))
    return items


def get_persona(persona_id: str) -> Optional[dict]:
    return _load_all().get(persona_id)


def build_system_prompt(data: dict) -> str:
    """YAML の system_prompt フィールドをそのまま返す。なければ role + description から生成。"""
    if "system_prompt" in data:
        return data["system_prompt"]
    name = data.get("name", "AI")
    role = data.get("role", "")
    description = data.get("description", "")
    return f"You are {name}. {role}. {description}"
