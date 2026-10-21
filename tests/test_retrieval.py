"""Tests de la recherche dans la base de connaissances et de son intégration."""

import pytest

from src.analyzer import analyze_message
from src.classifier import classify_message
from src.knowledge_base import MIN_RELIABLE_SCORE, load_knowledge_base
from src.models import Category

KB = load_knowledge_base()


def _top_id(text: str) -> str | None:
    category = classify_message(text).category
    results = KB.search(text, category=category, min_score=MIN_RELIABLE_SCORE)
    return results[0].doc_id if results else None


@pytest.mark.parametrize(
    "text, expected_id",
    [
        ("How long does standard delivery take?", "kb-001"),
        ("Where is my order ORD-12345?", "kb-002"),
        ("My parcel never arrived", "kb-003"),
        ("What is your return policy?", "kb-004"),
        ("How do I get a refund?", "kb-005"),
        ("My package arrived damaged", "kb-006"),
        ("I forgot my password", "kb-007"),
        ("My earbuds are not turning on", "kb-008"),
        ("Do you have NovaBuds in white?", "kb-010"),
        ("Quels sont vos horaires ?", "kb-013"),
        ("Quel est le délai de livraison ?", "kb-001"),
    ],
)
def test_best_entry_is_found(text, expected_id):
    assert _top_id(text) == expected_id


@pytest.mark.parametrize(
    "text",
    ["Do you ship to Mars?", "Tell me a joke", "I like apples", "Where are you located?"],
)
def test_unrelated_questions_find_nothing(text):
    assert KB.search(text) == []


def test_results_are_sorted_and_limited():
    results = KB.search("delivery tracking order status", top_k=2)
    assert len(results) <= 2
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_category_bonus_raises_score_but_cannot_create_a_match():
    plain = KB.search("return")[0].score
    boosted = KB.search("return", category=Category.REFUNDS_RETURNS)[0].score
    assert boosted > plain
    assert KB.search("zzz qqq", category=Category.REFUNDS_RETURNS) == []


def test_search_edge_cases():
    assert KB.search("delivery", top_k=0) == []
    assert KB.search("   ") == []
    assert KB.search("delivery", min_score=1.1) == []


def test_whole_word_keywords_avoid_false_positives():
    # "app!" ne doit pas correspondre à "apple".
    assert all(r.doc_id != "kb-009" for r in KB.search("I bought an apple"))


# --- Intégration dans l'analyse complète ------------------------------------
def test_answerable_question_is_grounded_and_not_escalated():
    result = analyze_message("How long does standard delivery take?")
    assert result.needs_human is False
    assert result.retrieved_information[0].doc_id == "kb-001"
    assert "3 to 5 business days" in result.suggested_response


def test_french_question_gets_french_intro():
    result = analyze_message("Quel est le délai de livraison ?")
    assert result.language == "fr"
    assert result.needs_human is False
    assert "Merci" in result.suggested_response
    assert "3 to 5 business days" in result.suggested_response


def test_unknown_question_is_escalated_without_inventing():
    result = analyze_message("Do you ship to Mars?")
    assert result.needs_human is True
    assert result.retrieved_information == []
    assert "reliable information" in result.escalation_reason
    assert "business days" not in result.suggested_response


def test_sensitive_case_is_escalated_but_keeps_context_for_human():
    result = analyze_message("I want a refund for my damaged item")
    assert result.needs_human is True
    assert "authorization" in result.escalation_reason
    assert result.retrieved_information  # visible pour l'agent humain
    # La réponse n'affirme rien sur le remboursement.
    assert "refund" not in result.suggested_response.lower()


def test_explicit_human_request_does_not_add_missing_info_reason():
    result = analyze_message("I want to talk to a human please")
    assert result.needs_human is True
    assert "reliable information" not in result.escalation_reason


def test_only_reliable_entries_are_returned():
    result = analyze_message("How long does standard delivery take?")
    assert all(r.score >= MIN_RELIABLE_SCORE for r in result.retrieved_information)
