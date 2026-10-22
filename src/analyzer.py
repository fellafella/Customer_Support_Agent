"""Assemble classification + recherche + escalade + réponse (LLM optionnel)."""

from __future__ import annotations

import logging

from src.classifier import classify_message
from src.escalation import evaluate_escalation
from src.knowledge_base import MIN_RELIABLE_SCORE, KnowledgeBase, load_knowledge_base
from src.llm_client import LLMClient, LLMError
from src.models import AnalysisResult, CustomerMessage
from src.response_generator import (
    build_fallback_response,
    build_grounded_response,
    generate_llm_reply,
)

logger = logging.getLogger(__name__)

_default_kb: KnowledgeBase | None = None


def _get_default_kb() -> KnowledgeBase:
    """Charge la base par défaut une seule fois (chargement paresseux)."""
    global _default_kb
    if _default_kb is None:
        _default_kb = load_knowledge_base()
    return _default_kb


def analyze_message(
    text: str,
    *,
    kb: KnowledgeBase | None = None,
    llm: LLMClient | None = None,
    previous_negative_messages: int = 0,
) -> AnalysisResult:
    """Analyse un message client et propose une réponse (jamais envoyée).

    Étapes : classification, recherche dans la base, décision d'escalade,
    puis réponse. Le LLM n'est appelé que si la demande n'est PAS escaladée
    par les règles : un cas sensible n'est jamais envoyé au LLM et l'IA ne
    tranche jamais seule. En cas d'erreur du LLM ou de réponse non conforme,
    on retombe sur la réponse de secours (la fiche citée telle quelle).

    Raises:
        pydantic.ValidationError: si le message est vide ou trop long.
    """
    message = CustomerMessage(text=text)
    knowledge_base = kb if kb is not None else _get_default_kb()
    classification = classify_message(message.text)

    retrieved = knowledge_base.search(
        message.text,
        category=classification.category,
        min_score=MIN_RELIABLE_SCORE,
    )

    # 1) Règles de sécurité/humain/action, indépendantes de la base.
    decision = evaluate_escalation(
        message.text,
        classification,
        previous_negative_messages=previous_negative_messages,
    )
    # 2) Si rien d'autre n'escalade, on regarde si l'information est fiable.
    if not decision.needs_human:
        decision = evaluate_escalation(
            message.text,
            classification,
            has_reliable_info=bool(retrieved),
            previous_negative_messages=previous_negative_messages,
        )

    response_source = "template"
    if decision.needs_human:
        suggested = build_fallback_response(classification.language, needs_human=True)
    else:
        suggested = build_grounded_response(classification.language, retrieved[0])
        if llm is not None:
            try:
                reply = generate_llm_reply(
                    llm, message.text, classification.language, retrieved
                )
            except LLMError as exc:
                # Pas de contenu client dans le log, seulement la cause.
                logger.warning("LLM non utilisé, réponse de secours : %s", exc)
            else:
                if reply.answerable:
                    suggested, response_source = reply.reply, "llm"
                else:
                    # Le LLM juge les fiches insuffisantes : on escalade.
                    decision = evaluate_escalation(
                        message.text,
                        classification,
                        has_reliable_info=False,
                        previous_negative_messages=previous_negative_messages,
                    )
                    suggested = build_fallback_response(
                        classification.language, needs_human=True
                    )

    return AnalysisResult(
        intent=classification.intent,
        category=classification.category,
        priority=classification.priority,
        sentiment=classification.sentiment,
        needs_human=decision.needs_human,
        escalation_reason=decision.reason,
        retrieved_information=retrieved,
        suggested_response=suggested,
        order_number=classification.order_number,
        language=classification.language,
        response_source=response_source,
    )
