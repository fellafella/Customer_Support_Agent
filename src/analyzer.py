"""Assemble classification + escalade en une analyse validée (AnalysisResult)."""

from __future__ import annotations

from src.classifier import classify_message
from src.escalation import evaluate_escalation
from src.models import AnalysisResult, CustomerMessage
from src.response_generator import build_fallback_response


def analyze_message(text: str, *, previous_negative_messages: int = 0) -> AnalysisResult:
    """Analyse un message client (Phase 1 : règles uniquement, sans LLM).

    Raises:
        pydantic.ValidationError: si le message est vide ou trop long.
    """
    message = CustomerMessage(text=text)
    classification = classify_message(message.text)
    decision = evaluate_escalation(
        message.text,
        classification,
        previous_negative_messages=previous_negative_messages,
    )
    return AnalysisResult(
        intent=classification.intent,
        category=classification.category,
        priority=classification.priority,
        sentiment=classification.sentiment,
        needs_human=decision.needs_human,
        escalation_reason=decision.reason,
        retrieved_information=[],  # Phase 2
        suggested_response=build_fallback_response(
            classification.language, decision.needs_human
        ),
        order_number=classification.order_number,
        language=classification.language,
    )
