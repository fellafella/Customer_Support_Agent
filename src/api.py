"""API locale de l'agent, appelée par le workflow n8n (Phase 6).

Lancement (depuis la racine du projet, environnement virtuel activé) :
    python -m uvicorn src.api:create_app --factory --host 127.0.0.1 --port 8000

Routes :
- GET  /health            : l'API répond (sans clé).
- POST /analyze           : analyse un message ; ``route`` dit à n8n où aller.
- POST /escalations       : consigne un cas escaladé (données sensibles masquées).
- POST /approved-replies  : enregistre une réponse approuvée par un humain.
                            C'est un ENVOI SIMULÉ : rien n'est envoyé à un client.

Sécurité :
- si SUPPORT_API_KEY est définie, toutes les routes sauf /health exigent
  l'en-tête ``X-API-Key`` ; sans clé configurée, l'API reste ouverte aux appels
  locaux : n'écoutez jamais sur 0.0.0.0 sans clé ;
- les textes consignés sont expurgés (e-mails, cartes, téléphones) ;
- les fichiers de consignation vivent dans data/runtime/ (exclu de Git).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.analyzer import analyze_message
from src.config import PROJECT_ROOT, load_settings
from src.handoff import format_handoff, redact_sensitive
from src.knowledge_base import KnowledgeBase, load_knowledge_base
from src.llm_client import LLMClient, create_llm_client
from src.logging_config import configure_logging
from src.models import AnalysisResult, HandoffSummary
from src.ui_helpers import draft_warnings

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 5000
MAX_REPLY_LENGTH = 3000
MAX_HISTORY = 20
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"

ROUTE_ESCALATE = "escalate_to_human"
ROUTE_APPROVAL = "await_human_approval"

_FROM_ENV: Any = object()  # valeur par défaut : « lire la configuration »


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    history: list[Annotated[str, Field(max_length=MAX_MESSAGE_LENGTH)]] = Field(
        default_factory=list, max_length=MAX_HISTORY
    )

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Le message ne peut pas être vide.")
        return value


class AnalyzeResponse(AnalysisResult):
    """Analyse complète, plus la route à suivre et le résumé déjà mis en texte."""

    route: Literal["escalate_to_human", "await_human_approval"]
    handoff_text: str | None = None


class EscalationRequest(BaseModel):
    customer_message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    handoff: HandoffSummary


class ApprovedReplyRequest(BaseModel):
    approved: Literal[True]  # seule une réponse approuvée peut être enregistrée
    approved_by: str = Field(..., min_length=1, max_length=80)
    customer_message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    reply: str = Field(..., min_length=1, max_length=MAX_REPLY_LENGTH)

    @field_validator("approved_by", "reply")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Ce champ ne peut pas être vide.")
        return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact_tree(value: Any) -> Any:
    """Masque les données sensibles dans tous les textes d'une structure JSON."""
    if isinstance(value, str):
        return redact_sensitive(value)
    if isinstance(value, list):
        return [_redact_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact_tree(v) for k, v in value.items()}
    return value


class _JsonlLog:
    """Fichier JSON Lines à écriture protégée par un verrou."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def append(self, entry: dict) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def create_app(
    *,
    kb: KnowledgeBase | None = None,
    llm: LLMClient | None = _FROM_ENV,
    runtime_dir: Path | str | None = None,
    api_key: str | None = _FROM_ENV,
) -> FastAPI:
    """Construit l'application.

    Args:
        kb: base de connaissances (par défaut, celle du projet).
        llm: client LLM. Absent : créé depuis .env si une clé existe.
            ``None`` : aucun LLM (réponses de secours).
        runtime_dir: dossier des fichiers de consignation.
        api_key: clé attendue dans ``X-API-Key``. Absente : lue dans
            SUPPORT_API_KEY. ``None`` : pas d'authentification.
    """
    configure_logging()
    if llm is _FROM_ENV or api_key is _FROM_ENV:
        load_dotenv(PROJECT_ROOT / ".env")
    if llm is _FROM_ENV:
        llm = create_llm_client(load_settings())
    if api_key is _FROM_ENV:
        api_key = (os.environ.get("SUPPORT_API_KEY") or "").strip() or None

    knowledge_base = kb or load_knowledge_base()
    folder = Path(runtime_dir) if runtime_dir else DEFAULT_RUNTIME_DIR
    escalations_log = _JsonlLog(folder / "escalations.jsonl")
    outbox_log = _JsonlLog(folder / "outbox_simulation.jsonl")

    app = FastAPI(title="AI Customer Support Agent API", version="0.6.0")

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if api_key is None:
            return
        if not x_api_key or not secrets.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    protected = [Depends(require_key)]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm": llm is not None, "auth": api_key is not None}

    @app.post("/analyze", response_model=AnalyzeResponse, dependencies=protected)
    def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
        result = analyze_message(
            body.message, kb=knowledge_base, llm=llm, history=body.history
        )
        return AnalyzeResponse(
            **result.model_dump(),
            route=ROUTE_ESCALATE if result.needs_human else ROUTE_APPROVAL,
            handoff_text=format_handoff(result.handoff) if result.handoff else None,
        )

    @app.post("/escalations", dependencies=protected)
    def record_escalation(body: EscalationRequest) -> dict:
        escalations_log.append(
            {
                "recorded_at": _now(),
                "customer_message": redact_sensitive(body.customer_message),
                "handoff": _redact_tree(body.handoff.model_dump(mode="json")),
            }
        )
        logger.info("Escalade consignée (triggers=%s)", ",".join(body.handoff.triggers) or "-")
        return {"status": "recorded"}

    @app.post("/approved-replies", dependencies=protected)
    def record_approved_reply(body: ApprovedReplyRequest) -> dict:
        warnings = draft_warnings(body.reply)
        outbox_log.append(
            {
                "recorded_at": _now(),
                "approved_by": body.approved_by,
                "customer_message": redact_sensitive(body.customer_message),
                "reply": redact_sensitive(body.reply),
                "warnings": warnings,
                "sent": False,
            }
        )
        logger.info("Réponse approuvée consignée (envoi simulé, avertissements=%d)", len(warnings))
        return {"status": "recorded", "sent": False, "simulated": True, "warnings": warnings}

    return app
