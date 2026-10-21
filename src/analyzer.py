"""Assemble classification + recherche + escalade en une analyse validée."""

from __future__ import annotations

from src.classifier import classify_message
from src.escalation import evaluate_escalation
from src.knowledge_base import MIN_RELIABLE_SCORE, KnowledgeBase, load_knowledge_base
from src.models import AnalysisResult, CustomerMessage
from src.response_generator import build_fallback_response, build_grounded_response

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
    previous_negative_messages: int = 0,
) -> AnalysisResult:
    """Analyse un message client (Phase 2 : règles + recherche, sans LLM).

    Étapes : classification, recherche dans la base, décision d'escalade,
    réponse suggérée. Seules les fiches fiables (score >= MIN_RELIABLE_SCORE)
    sont conservées.

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
    has_reliable_info = bool(retrieved)

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
            has_reliable_info=has_reliable_info,
            previous_negative_messages=previous_negative_messages,
        )

    if decision.needs_human:
        # L'IA ne tranche pas : réponse d'attente, la fiche reste visible
        # pour l'agent humain.
        suggested = build_fallback_response(classification.language, needs_human=True)
    else:
        suggested = build_grounded_response(classification.language, retrieved[0])

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
    )
