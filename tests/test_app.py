"""Tests de l'interface Streamlit (sans navigateur, via AppTest) et des utilitaires."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.models import Priority
from src.models import AnalysisResult, Category, Sentiment
from src.ui_helpers import (
    SAMPLE_MESSAGES,
    draft_warnings,
    empty_state_html,
    masthead_html,
    priority_markdown,
    relevance_bar_html,
    ticket_html,
)

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch):
    """Lance l'application SANS LLM, même si un vrai .env existe."""
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=15).run()


def _analyze(app, text):
    app.text_area(key="message_input").set_value(text)
    app.button(key="analyze").click()
    return app.run()


def _texts(elements):
    return " ".join(str(e.value) for e in elements)


def _html(app):
    """Tout le HTML affiché (sans la feuille de style)."""
    return " ".join(
        str(e.value) for e in app.get("html") if not str(e.value).startswith("<style>")
    )


# --- Utilitaires ---------------------------------------------------------------
def test_priority_markdown_has_color_and_label():
    assert priority_markdown(Priority.URGENT) == ":red[**URGENT**]"
    assert priority_markdown(Priority.LOW) == ":green[**LOW**]"


def test_draft_warnings():
    assert draft_warnings("Standard delivery takes 3 to 5 days.") == []
    assert draft_warnings("   ")
    assert draft_warnings("Your refund has been approved.")
    assert draft_warnings("Nous avons annulé votre commande.")


def test_sample_messages_are_not_empty():
    assert len(SAMPLE_MESSAGES) >= 5 and all(m.strip() for m in SAMPLE_MESSAGES)


# --- Interface -------------------------------------------------------------------
def test_app_starts_without_error(app):
    assert not app.exception
    assert "Customer support desk" in _html(app)
    assert "No case yet" in _html(app)
    assert app.button(key="analyze") is not None
    assert "No LLM configured" in _texts(app.sidebar.info)


def test_empty_message_shows_warning(app):
    app = _analyze(app, "   ")
    assert "Enter a customer message first." in _texts(app.warning)
    assert app.session_state.result is None


def test_ordinary_question_is_not_escalated(app):
    app = _analyze(app, "How long does standard delivery take?")
    assert not app.exception
    assert "Draft ready" in _html(app)
    assert "No rule asks for a human" in _html(app)
    draft = app.text_area(key=f"draft_{app.session_state.analysis_id}")
    assert "3 to 5 business days" in draft.value
    assert any("kb-001" in e.label for e in app.expander)


def test_sensitive_message_shows_escalation_and_summary(app):
    app = _analyze(app, "Someone made an unauthorized payment with my card, this is urgent!")
    assert "Needs a human" in _html(app)
    assert "prio-urgent" in _html(app)
    assert app.session_state.result.handoff is not None
    assert any("Case summary" in t.label for t in app.tabs)
    assert "holding message" in _texts(app.warning)


def test_unknown_question_says_nothing_was_invented(app):
    app = _analyze(app, "Do you ship to Mars?")
    assert "Needs a human" in _html(app)
    assert "Nothing was invented" in _texts(app.info)


def test_approve_saves_edited_draft_and_never_sends(app):
    app = _analyze(app, "How long does standard delivery take?")
    key = f"draft_{app.session_state.analysis_id}"
    app.text_area(key=key).set_value("Hello! Delivery takes 3 to 5 business days.")
    app.button(key="approve").click()
    app = app.run()
    approved = app.session_state.approved
    assert len(approved) == 1
    assert approved[0]["reply"] == "Hello! Delivery takes 3 to 5 business days."
    assert "NOT been sent" in _texts(app.success)
    assert app.button(key="approve").disabled is True


def test_draft_with_promise_shows_warning(app):
    app = _analyze(app, "How long does standard delivery take?")
    key = f"draft_{app.session_state.analysis_id}"
    app.text_area(key=key).set_value("Your refund has been approved.")
    app = app.run()
    assert "promise or confirm an action" in _texts(app.warning)


def test_conversation_history_triggers_persistent_dissatisfaction(app):
    app = _analyze(app, "My order ORD-77001 is late, this is unacceptable")
    app = _analyze(app, "Still nothing and this is terrible")
    assert len(app.session_state.history) == 2
    assert "remains dissatisfied" in _html(app)


def test_new_conversation_resets_state(app):
    app = _analyze(app, "How long does standard delivery take?")
    app.button(key="new_conversation").click()
    app = app.run()
    assert app.session_state.history == []
    assert app.session_state.result is None


def test_html_in_customer_message_is_not_rendered(app):
    app = _analyze(app, "<script>alert(1)</script> I want a human")
    assert not app.exception
    # Le message est affiché via st.text (pas de HTML interprété).
    assert any("<script>" in str(t.value) for t in app.text)


def test_style_sheet_is_injected(app):
    styles = [str(e.value) for e in app.get("html") if str(e.value).startswith("<style>")]
    assert styles and ".ticket" in styles[0]


def test_toast_is_shown_after_analysis(app):
    app = _analyze(app, "I want to talk to a human please")
    assert any("needs a human" in str(t.value).lower() for t in app.toast)


# --- HTML builders ---------------------------------------------------------------
def _result(**overrides):
    data = dict(
        intent="report_late_or_lost_delivery",
        category=Category.ORDERS_DELIVERY,
        priority=Priority.HIGH,
        sentiment=Sentiment.NEGATIVE,
        needs_human=False,
    )
    data.update(overrides)
    return AnalysisResult(**data)


def test_ticket_escapes_html_everywhere():
    evil = '<img src=x onerror="alert(1)">'
    html = ticket_html(
        _result(needs_human=True, escalation_reason=evil, order_number=evil), case_number=7
    )
    assert "<img" not in html
    assert "&lt;img" in html


def test_ticket_reflects_priority_and_status():
    ok = ticket_html(_result(), case_number=3)
    assert "prio-high" in ok and "Draft ready" in ok and "Case 003" in ok
    alert = ticket_html(_result(needs_human=True, escalation_reason="Because."), case_number=4)
    assert "Needs a human" in alert and "Because." in alert


def test_ticket_trail_has_four_steps():
    assert ticket_html(_result(), case_number=1).count('class="step"') == 4


def test_relevance_bar_is_clamped():
    assert "--w:100%" in relevance_bar_html(7)
    assert "--w:0%" in relevance_bar_html(-1)
    assert "--w:50%" in relevance_bar_html(0.5)


def test_static_html_blocks_are_not_empty():
    assert "Customer support desk" in masthead_html()
    assert "No case yet" in empty_state_html()
