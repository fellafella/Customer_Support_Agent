"""Résumé de cas pour l'agent humain (passage de relais).

Tout est construit par des règles, sans LLM : le résumé est déterministe,
ne dépend d'aucun service externe et n'envoie aucune donnée supplémentaire à
un tiers. Les données sensibles du client (e-mail, carte, téléphone) sont
masquées avant d'être recopiées dans le résumé.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.classifier import Classification
from src.escalation import (
    ACTION_REQUIRED,
    FRAUD_OR_SECURITY,
    HUMAN_REQUESTED,
    NO_RELIABLE_INFO,
    PERSISTENT_DISSATISFACTION,
    SENSITIVE_BILLING,
)
from src.models import EscalationDecision, HandoffSummary, RetrievedInformation

EXCERPT_LENGTH = 300
MAX_MESSAGES_SHOWN = 5

AI_NOTE = (
    "The AI performed no action and made no promise to the customer. "
    "The customer only received a holding message."
)

RECOMMENDED_ACTIONS: dict[str, str] = {
    HUMAN_REQUESTED: "The customer asked for a human: reply personally and promptly.",
    FRAUD_OR_SECURITY: (
        "Treat as a security/privacy incident: follow the fraud procedure and verify "
        "the customer's identity through an official channel. Never ask for a full "
        "card number or password."
    ),
    SENSITIVE_BILLING: (
        "Check the payment records before replying; involve billing or legal "
        "if the customer mentions a dispute or a lawyer."
    ),
    ACTION_REQUIRED: (
        "Verify the order and the customer's eligibility, then decide and perform "
        "the requested action in the order system. The AI did NOT do it."
    ),
    NO_RELIABLE_INFO: (
        "Answer manually: the knowledge base had no reliable answer. "
        "Consider adding this topic to the knowledge base."
    ),
    PERSISTENT_DISSATISFACTION: (
        "The customer has been unhappy over several messages: apologise, "
        "take ownership and resolve the issue personally."
    ),
}

_CATEGORY_LABELS = {
    "product_information": "Product information",
    "orders_and_delivery": "Orders and delivery",
    "refunds_and_returns": "Refunds and returns",
    "technical_support": "Technical support",
    "complaint": "Complaint",
    "other": "Other",
}

# --- Masquage des données sensibles ----------------------------------------
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){13,19}(?![\w-])")
_PHONE = re.compile(r"(?<![\w-])(?:\+\d[\d ().-]{7,}\d|\d{2,4}(?:[ .-]\d{2,4}){2,4})(?![\w-])")


def redact_sensitive(text: str) -> str:
    """Remplace e-mails, numéros de carte et téléphones par des marqueurs.

    Les numéros de commande (ex. ORD-12345) sont conservés : l'agent en a besoin.
    """
    text = _EMAIL.sub("[email hidden]", text)
    text = _CARD.sub("[card number hidden]", text)
    return _PHONE.sub("[phone hidden]", text)


def _excerpt(text: str, limit: int = EXCERPT_LENGTH) -> str:
    """Nettoie, masque et tronque un message pour l'affichage."""
    clean = re.sub(r"\s+", " ", redact_sensitive(text)).strip()
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


def build_handoff_summary(
    *,
    customer_messages: Sequence[str],
    classification: Classification,
    decision: EscalationDecision,
    retrieved: Sequence[RetrievedInformation] = (),
    previous_negative_messages: int = 0,
) -> HandoffSummary:
    """Construit le résumé de cas d'une conversation escaladée.

    Args:
        customer_messages: messages du client dans l'ordre, le dernier en dernier.
        classification: classification du dernier message.
        decision: décision d'escalade (doit demander un humain).
        retrieved: fiches de la base trouvées (utiles à l'agent).
        previous_negative_messages: nombre de messages négatifs précédents.

    Raises:
        ValueError: si aucune escalade n'est demandée ou si la liste est vide.
    """
    if not decision.needs_human or not decision.reason:
        raise ValueError("Un résumé n'est construit que pour une conversation escaladée.")
    if not customer_messages:
        raise ValueError("Au moins un message client est nécessaire.")

    shown = [_excerpt(m) for m in customer_messages[-MAX_MESSAGES_SHOWN:]]
    count = len(customer_messages)
    summary = (
        f"{count} customer message{'s' if count > 1 else ''}. "
        f"Sentiment of the latest message: {classification.sentiment.value}. "
        f"Latest message: \"{shown[-1]}\""
    )
    if previous_negative_messages:
        summary += (
            f" The customer was already unhappy in {previous_negative_messages} "
            f"earlier message{'s' if previous_negative_messages > 1 else ''}."
        )

    actions = [RECOMMENDED_ACTIONS[t] for t in decision.triggers if t in RECOMMENDED_ACTIONS]
    category_label = _CATEGORY_LABELS[classification.category.value]
    return HandoffSummary(
        headline=(
            f"[{classification.priority.value.upper()}] {category_label} - "
            f"{classification.intent.replace('_', ' ')}"
        ),
        priority=classification.priority,
        category=classification.category,
        intent=classification.intent,
        sentiment=classification.sentiment,
        customer_language=classification.language,
        order_number=classification.order_number,
        escalation_reason=decision.reason,
        triggers=list(decision.triggers),
        conversation_summary=summary,
        customer_messages=shown,
        relevant_kb=[f"{info.doc_id} - {info.title}" for info in retrieved],
        recommended_actions=actions,
        ai_actions_taken=[],
        ai_note=AI_NOTE,
    )


def format_handoff(summary: HandoffSummary) -> str:
    """Met le résumé en texte lisible (console, ticket, e-mail interne)."""
    lines = [
        summary.headline,
        "-" * len(summary.headline),
        f"Order number : {summary.order_number or 'not provided'}",
        f"Language     : {summary.customer_language}",
        f"Sentiment    : {summary.sentiment.value}",
        "",
        "Why this case is escalated:",
        f"  {summary.escalation_reason}",
        "",
        "Conversation:",
        f"  {summary.conversation_summary}",
    ]
    if summary.relevant_kb:
        lines += ["", "Relevant knowledge base entries:"]
        lines += [f"  - {item}" for item in summary.relevant_kb]
    if summary.recommended_actions:
        lines += ["", "Recommended next steps:"]
        lines += [f"  - {action}" for action in summary.recommended_actions]
    lines += ["", f"AI note: {summary.ai_note}"]
    return "\n".join(lines)
