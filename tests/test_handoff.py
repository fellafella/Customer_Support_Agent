"""Tests de la Phase 4 : passage à l'humain et résumé de cas."""

import pytest
from pydantic import ValidationError

from src.analyzer import analyze_message
from src.classifier import classify_message
from src.escalation import (
    ACTION_REQUIRED,
    FRAUD_OR_SECURITY,
    NO_RELIABLE_INFO,
    PERSISTENT_DISSATISFACTION,
    evaluate_escalation,
)
from src.handoff import (
    AI_NOTE,
    build_handoff_summary,
    format_handoff,
    redact_sensitive,
)
from src.models import AnalysisResult, Category, EscalationDecision, Priority, Sentiment
from src.response_generator import build_fallback_response


class SpyLLM:
    """Faux LLM qui enregistre les appels ; ne doit jamais servir aux cas escaladés."""

    def __init__(self):
        self.calls = 0

    def generate_structured(self, *, system_instruction, prompt, schema):
        self.calls += 1
        raise AssertionError("Le LLM ne doit pas être appelé pour un cas escaladé.")


# --- Masquage des données sensibles ------------------------------------------
@pytest.mark.parametrize(
    "text, hidden, kept",
    [
        ("Write to jean.dupont@mail.com please", "[email hidden]", "jean.dupont@mail.com"),
        ("My card is 4111 1111 1111 1111", "[card number hidden]", "4111"),
        ("Card 4111-1111-1111-1111 was used", "[card number hidden]", "4111-1111"),
        ("Call me on +33 6 12 34 56 78", "[phone hidden]", "12 34 56"),
        ("Mon numéro : 06 12 34 56 78", "[phone hidden]", "12 34 56"),
    ],
)
def test_sensitive_data_is_redacted(text, hidden, kept):
    result = redact_sensitive(text)
    assert hidden in result
    assert kept not in result


@pytest.mark.parametrize(
    "text", ["Order ORD-12345 is late", "Commande 55120 cassée", "It takes 3 to 5 days"]
)
def test_order_numbers_and_ordinary_text_are_kept(text):
    assert redact_sensitive(text) == text


# --- Résumé de cas -----------------------------------------------------------
def _escalated(text, **kwargs):
    return analyze_message(text, **kwargs)


def test_fraud_case_summary():
    result = _escalated("Fraud! Card 4111 1111 1111 1111 was used, order ORD-10001. Urgent!")
    summary = result.handoff
    assert result.needs_human is True and summary is not None
    assert summary.priority == Priority.URGENT
    assert summary.headline.startswith("[URGENT]")
    assert summary.order_number == "ORD-10001"
    assert FRAUD_OR_SECURITY in summary.triggers
    assert "4111" not in " ".join(summary.customer_messages)
    assert "[card number hidden]" in summary.conversation_summary
    assert any("full card number" in a for a in summary.recommended_actions)
    assert summary.escalation_reason == result.escalation_reason


def test_action_case_states_the_ai_did_nothing():
    summary = _escalated("I want a refund for my damaged item ORD-20002").handoff
    assert ACTION_REQUIRED in summary.triggers
    assert summary.ai_actions_taken == []
    assert summary.ai_note == AI_NOTE
    assert any("AI did NOT do it" in a for a in summary.recommended_actions)
    assert any(item.startswith("kb-") for item in summary.relevant_kb)


def test_no_reliable_info_case_suggests_manual_answer():
    summary = _escalated("Do you ship to Mars?").handoff
    assert NO_RELIABLE_INFO in summary.triggers
    assert any("manually" in a for a in summary.recommended_actions)


def test_non_escalated_case_has_no_handoff():
    result = analyze_message("How long does standard delivery take?")
    assert result.needs_human is False and result.handoff is None


def test_long_messages_are_truncated_and_cleaned():
    text = "human " + "very long text " * 100
    summary = _escalated(text).handoff
    assert len(summary.customer_messages[-1]) <= 300
    assert summary.customer_messages[-1].endswith("…")


def test_only_last_messages_are_shown():
    history = [f"message number {i}" for i in range(8)]
    summary = _escalated("I want to talk to a human", history=history).handoff
    assert len(summary.customer_messages) == 5
    assert summary.customer_messages[-1] == "I want to talk to a human"
    assert summary.conversation_summary.startswith("9 customer messages.")


def test_french_customer_is_flagged_in_summary():
    summary = _escalated("Je veux parler à un conseiller, commande ORD-30003").handoff
    assert summary.customer_language == "fr"
    assert summary.order_number == "ORD-30003"


def test_build_summary_refuses_non_escalated_decision():
    classification = classify_message("How long does delivery take?")
    with pytest.raises(ValueError):
        build_handoff_summary(
            customer_messages=["x"],
            classification=classification,
            decision=EscalationDecision(needs_human=False),
        )


def test_build_summary_refuses_empty_conversation():
    classification = classify_message("I want a human")
    decision = evaluate_escalation("I want a human", classification)
    with pytest.raises(ValueError):
        build_handoff_summary(
            customer_messages=[], classification=classification, decision=decision
        )


def test_format_handoff_is_readable():
    text = format_handoff(_escalated("Someone hacked my account, order ORD-40004").handoff)
    for expected in (
        "Why this case is escalated:",
        "Conversation:",
        "Recommended next steps:",
        "AI note:",
        "ORD-40004",
    ):
        assert expected in text


# --- Historique de conversation ----------------------------------------------
def test_persistent_dissatisfaction_uses_history():
    follow_up = "Still nothing and this is terrible"
    result = _escalated(follow_up, history=["My order is late, this is unacceptable"])
    assert result.needs_human is True
    assert PERSISTENT_DISSATISFACTION in result.handoff.triggers
    assert "already unhappy in 1 earlier message" in result.handoff.conversation_summary


def test_order_number_from_earlier_message_is_kept():
    result = _escalated(
        "This is terrible, still nothing", history=["Order ORD-77001 is late, unacceptable"]
    )
    assert result.order_number == "ORD-77001"
    assert result.handoff.order_number == "ORD-77001"


# --- Le LLM ne résout jamais un cas qui demande une autorisation --------------
@pytest.mark.parametrize(
    "text",
    [
        "I want a refund",
        "Please cancel my order ORD-50005",
        "I want to cancel it",
        "Je veux annuler ma commande",
        "Can you change my address?",
        "Remboursez-moi immédiatement",
        "Please delete my account",
        "Someone hacked my account",
        "There is an unauthorized charge on my card",
        "I was charged twice, I will call my lawyer",
        "Let me speak to a manager",
        "Do you ship to Mars?",
    ],
)
def test_escalated_cases_are_never_resolved_by_the_ai(text):
    llm = SpyLLM()
    result = analyze_message(text, llm=llm)
    assert result.needs_human is True
    assert llm.calls == 0
    assert result.response_source == "template"
    assert result.suggested_response == build_fallback_response(result.language, True)
    assert result.handoff is not None and result.handoff.ai_actions_taken == []


def test_cancel_intent_alone_triggers_action_rule():
    decision = evaluate_escalation("I want to cancel it", classify_message("I want to cancel it"))
    assert ACTION_REQUIRED in decision.triggers


# --- Cohérence du modèle -----------------------------------------------------
def test_handoff_forbidden_without_escalation():
    summary = _escalated("I want a human").handoff
    with pytest.raises(ValidationError):
        AnalysisResult(
            intent="x",
            category=Category.OTHER,
            priority=Priority.LOW,
            sentiment=Sentiment.NEUTRAL,
            needs_human=False,
            handoff=summary,
        )
