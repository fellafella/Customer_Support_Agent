"""Tests des règles d'escalade et du pipeline d'analyse complet."""

import pytest

from src.analyzer import analyze_message
from src.classifier import classify_message
from src.escalation import (
    ACTION_REQUIRED,
    FRAUD_OR_SECURITY,
    HUMAN_REQUESTED,
    NO_RELIABLE_INFO,
    PERSISTENT_DISSATISFACTION,
    SENSITIVE_BILLING,
    evaluate_escalation,
)


def _decide(text, **kwargs):
    return evaluate_escalation(text, classify_message(text), **kwargs)


@pytest.mark.parametrize(
    "text, trigger",
    [
        ("I want to speak to a human", HUMAN_REQUESTED),
        ("Je veux parler à un conseiller", HUMAN_REQUESTED),
        ("Someone hacked my account", FRAUD_OR_SECURITY),
        ("Il y a un prélèvement non autorisé", FRAUD_OR_SECURITY),
        ("I will call my lawyer and ask for a chargeback", SENSITIVE_BILLING),
        ("I was charged twice for my order", SENSITIVE_BILLING),
        ("Please cancel my order", ACTION_REQUIRED),
        ("Remboursez-moi immédiatement", ACTION_REQUIRED),
        ("I want a refund", ACTION_REQUIRED),
    ],
)
def test_escalation_triggers(text, trigger):
    decision = _decide(text)
    assert decision.needs_human is True
    assert trigger in decision.triggers
    assert decision.reason


@pytest.mark.parametrize(
    "text",
    [
        "How long does delivery take?",
        "What is your return policy?",
        "How do I get a refund?",
        "Do you ship to Canada?",
    ],
)
def test_ordinary_questions_do_not_escalate(text):
    decision = _decide(text)
    assert decision.needs_human is False
    assert decision.reason is None
    assert decision.triggers == []


def test_no_reliable_info_escalates():
    assert NO_RELIABLE_INFO in _decide("Do you ship to Mars?", has_reliable_info=False).triggers


def test_unknown_info_status_does_not_escalate():
    assert _decide("Do you ship to Mars?", has_reliable_info=None).needs_human is False


def test_persistent_dissatisfaction():
    text = "This is still terrible and I am frustrated"
    assert _decide(text, previous_negative_messages=0).needs_human is False
    decision = _decide(text, previous_negative_messages=1)
    assert PERSISTENT_DISSATISFACTION in decision.triggers


def test_multiple_triggers_are_combined():
    decision = _decide("This is fraud! Let me talk to a human")
    assert {FRAUD_OR_SECURITY, HUMAN_REQUESTED} <= set(decision.triggers)


# --- Pipeline complet -------------------------------------------------------
def test_analyze_ordinary_message():
    result = analyze_message("How long does standard delivery take?")
    assert result.needs_human is False
    assert result.escalation_reason is None
    assert result.suggested_response


def test_analyze_sensitive_message_in_french():
    result = analyze_message("Fraude sur ma carte, c'est urgent ! Commande ORD-10001")
    assert result.needs_human is True
    assert result.priority.value == "urgent"
    assert result.language == "fr"
    assert result.order_number == "ORD-10001"
    assert "Merci" in result.suggested_response


def test_fallback_response_never_promises_refund():
    result = analyze_message("I want a refund now")
    text = result.suggested_response.lower()
    assert result.needs_human is True
    assert "will be refunded" not in text
    assert "refund has been" not in text


def test_analyze_rejects_empty_message():
    with pytest.raises(Exception):
        analyze_message("   ")
