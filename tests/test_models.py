"""Tests des modèles Pydantic."""

import pytest
from pydantic import ValidationError

from src.models import (
    AnalysisResult,
    Category,
    CustomerMessage,
    EscalationDecision,
    Priority,
    Sentiment,
)


def _valid_kwargs(**overrides):
    data = dict(
        intent="track_order",
        category=Category.ORDERS_DELIVERY,
        priority=Priority.MEDIUM,
        sentiment=Sentiment.NEUTRAL,
        needs_human=False,
    )
    data.update(overrides)
    return data


def test_customer_message_is_stripped():
    assert CustomerMessage(text="  hello  ").text == "hello"


@pytest.mark.parametrize("bad", ["", "   ", "x" * 5001])
def test_customer_message_rejects_invalid_text(bad):
    with pytest.raises(ValidationError):
        CustomerMessage(text=bad)


def test_analysis_result_valid():
    result = AnalysisResult(**_valid_kwargs())
    assert result.retrieved_information == []
    assert result.escalation_reason is None


def test_analysis_rejects_unknown_priority():
    with pytest.raises(ValidationError):
        AnalysisResult(**_valid_kwargs(priority="super-urgent"))


def test_analysis_requires_reason_when_human_needed():
    with pytest.raises(ValidationError):
        AnalysisResult(**_valid_kwargs(needs_human=True))


def test_analysis_rejects_reason_without_escalation():
    with pytest.raises(ValidationError):
        AnalysisResult(**_valid_kwargs(escalation_reason="because"))


def test_analysis_rejects_unsupported_language():
    with pytest.raises(ValidationError):
        AnalysisResult(**_valid_kwargs(language="de"))


def test_escalation_decision_consistency():
    assert EscalationDecision(needs_human=False).reason is None
    with pytest.raises(ValidationError):
        EscalationDecision(needs_human=True)
