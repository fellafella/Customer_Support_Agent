"""Phase 7 : scénarios réalistes (normal, urgent, sensible) en français et en anglais."""

import pytest

from src.analyzer import analyze_message
from src.models import Priority

# (message, escalade attendue)
SCENARIOS = [
    # --- Questions ordinaires : l'agent répond avec une fiche ---
    ("How long does standard delivery take?", False),
    ("How can I track my parcel?", False),
    ("What is your return policy?", False),
    ("How do I get a refund?", False),
    ("I forgot my password", False),
    ("My earbuds are not turning on", False),
    ("Do you have NovaBuds in white?", False),
    ("What payment methods do you accept?", False),
    ("Quel est le délai de livraison ?", False),
    ("Comment suivre mon colis ?", False),
    ("Quels sont vos horaires ?", False),
    ("Quels moyens de paiement acceptez-vous ?", False),
    # --- Cas sensibles : toujours un humain ---
    ("Someone hacked my account", True),
    ("There is an unauthorized charge on my card", True),
    ("My personal data was leaked", True),
    ("Someone stole my identity", True),
    ("I was charged twice for my order", True),
    ("I will call my lawyer", True),
    ("I want a refund", True),
    ("Please cancel my order ORD-12345", True),
    ("Can you change my address?", True),
    ("Please delete my account", True),
    ("I want to speak to a manager", True),
    ("Je veux parler à un conseiller", True),
    ("Remboursez-moi immédiatement", True),
    ("Il y a un prélèvement non autorisé sur ma carte", True),
    ("Je veux annuler ma commande", True),
    # --- Hors base : pas d'information fiable, donc un humain ---
    ("Do you ship to Mars?", True),
    ("Tell me a joke", True),
]


@pytest.mark.parametrize("text, expected", SCENARIOS)
def test_escalation_decision(text, expected):
    result = analyze_message(text)
    assert result.needs_human is expected, result.escalation_reason
    if expected:
        assert result.handoff is not None
    else:
        assert result.retrieved_information


@pytest.mark.parametrize(
    "text",
    [
        "This is unacceptable, my order is late, I need it immediately",
        "Someone made an unauthorized payment with my card, this is urgent!",
        "C'est inacceptable, je veux ma commande immédiatement",
    ],
)
def test_urgent_cases_get_the_highest_priority(text):
    assert analyze_message(text).priority == Priority.URGENT


def test_a_calm_question_is_not_urgent():
    assert analyze_message("How long does standard delivery take?").priority != Priority.URGENT


def test_conversation_flow_escalates_when_the_customer_stays_unhappy():
    history = ["My order ORD-77001 is late, this is unacceptable"]
    result = analyze_message("Still nothing and this is terrible", history=history)
    assert result.needs_human is True
    assert result.order_number == "ORD-77001"
    assert "remains dissatisfied" in result.escalation_reason
