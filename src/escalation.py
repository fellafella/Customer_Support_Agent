"""Règles d'escalade vers un agent humain.

Principe de prudence : au moindre doute sur un cas sensible, on escalade.
Chaque règle est indépendante, ce qui la rend facile à tester et à faire évoluer.
"""

from __future__ import annotations

from src.classifier import Classification
from src.keywords import (
    ACTION_REQUIRED_KEYWORDS,
    FRAUD_SECURITY_KEYWORDS,
    HUMAN_REQUEST_KEYWORDS,
    SENSITIVE_BILLING_KEYWORDS,
    find_matches,
    normalize_text,
)
from src.models import EscalationDecision, Sentiment

# Codes de déclencheurs (utiles pour les tests, les logs et le tableau de bord).
HUMAN_REQUESTED = "human_requested"
FRAUD_OR_SECURITY = "fraud_or_security"
SENSITIVE_BILLING = "sensitive_billing_or_legal"
ACTION_REQUIRED = "action_requires_authorization"
NO_RELIABLE_INFO = "no_reliable_information"
PERSISTENT_DISSATISFACTION = "persistent_dissatisfaction"

REASONS: dict[str, str] = {
    HUMAN_REQUESTED: "The customer asked to speak with a human agent.",
    FRAUD_OR_SECURITY: "The message mentions fraud, a security issue or a privacy concern.",
    SENSITIVE_BILLING: "The message involves a sensitive billing, refund or legal matter.",
    ACTION_REQUIRED: (
        "The request requires an action or authorization "
        "(e.g. refund, cancellation, account change) that the AI cannot perform."
    ),
    NO_RELIABLE_INFO: "No reliable information was found in the knowledge base.",
    PERSISTENT_DISSATISFACTION: "The customer remains dissatisfied after previous messages.",
}


def evaluate_escalation(
    text: str,
    classification: Classification,
    *,
    has_reliable_info: bool | None = None,
    previous_negative_messages: int = 0,
) -> EscalationDecision:
    """Décide si la conversation doit être transférée à un humain.

    Args:
        text: message brut du client.
        classification: résultat de ``classify_message``.
        has_reliable_info: ``False`` si la base de connaissances n'a rien de
            fiable (renseigné à partir de la Phase 2) ; ``None`` = inconnu.
        previous_negative_messages: nombre de messages négatifs précédents
            dans la même conversation.

    Returns:
        Une ``EscalationDecision`` avec les déclencheurs et la raison.
    """
    normalized = normalize_text(text)
    triggers: list[str] = []

    if find_matches(normalized, HUMAN_REQUEST_KEYWORDS):
        triggers.append(HUMAN_REQUESTED)
    if find_matches(normalized, FRAUD_SECURITY_KEYWORDS):
        triggers.append(FRAUD_OR_SECURITY)
    if find_matches(normalized, SENSITIVE_BILLING_KEYWORDS):
        triggers.append(SENSITIVE_BILLING)
    if find_matches(normalized, ACTION_REQUIRED_KEYWORDS):
        triggers.append(ACTION_REQUIRED)
    if has_reliable_info is False:
        triggers.append(NO_RELIABLE_INFO)
    if classification.sentiment == Sentiment.NEGATIVE and previous_negative_messages >= 1:
        triggers.append(PERSISTENT_DISSATISFACTION)

    if not triggers:
        return EscalationDecision(needs_human=False)
    return EscalationDecision(
        needs_human=True,
        reason=" ".join(REASONS[t] for t in triggers),
        triggers=triggers,
    )