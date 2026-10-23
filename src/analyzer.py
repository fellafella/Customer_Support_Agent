"""Assemble classification + recherche + escalade + réponse (LLM optionnel)."""

from __future__ import annotations

import logging
from dataclasses import replace
from collections.abc import Sequence

from src.classifier import classify_message, extract_order_number
from src.escalation import evaluate_escalation
from src.knowledge_base import MIN_RELIABLE_SCORE, KnowledgeBase, load_knowledge_base
from src.handoff import build_handoff_summary
from src.llm_client import LLMClient, LLMError
from src.models import AnalysisResult, CustomerMessage, Sentiment
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
    history: Sequence[str] = (),
    previous_negative_messages: int = 0,
) -> AnalysisResult:
    """Analyse un message client et propose une réponse (jamais envoyée).

    ``history`` contient les messages précédents du même client (du plus
    ancien au plus récent) : ils servent à détecter un mécontentement qui
    dure et à rédiger le résumé de cas pour l'agent humain.

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
    if classification.order_number is None:
        # Le numéro de commande a pu être donné dans un message précédent.
        earlier = next(
            (n for n in map(extract_order_number, reversed(list(history))) if n), None
        )
        classification = replace(classification, order_number=earlier)
    previous_negative_messages = max(
        previous_negative_messages,
        sum(1 for old in history if classify_message(old).sentiment == Sentiment.NEGATIVE),
    )

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

    handoff = None
    if decision.needs_human:
        handoff = build_handoff_summary(
            customer_messages=[*history, message.text],
            classification=classification,
            decision=decision,
            retrieved=retrieved,
            previous_negative_messages=previous_negative_messages,
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
        handoff=handoff,
    )
