"""Tests de l'API locale appelée par n8n (sans réseau ni vrai LLM)."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.knowledge_base import load_knowledge_base

KEY = "test-key-123"
KB = load_knowledge_base()


@pytest.fixture
def runtime(tmp_path):
    return tmp_path / "runtime"


@pytest.fixture
def client(runtime):
    return TestClient(create_app(kb=KB, llm=None, runtime_dir=runtime, api_key=None))


@pytest.fixture
def secured(runtime):
    return TestClient(create_app(kb=KB, llm=None, runtime_dir=runtime, api_key=KEY))


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


APPROVED = {
    "approved": True,
    "approved_by": "Amina",
    "customer_message": "How long does standard delivery take?",
    "reply": "Standard delivery takes 3 to 5 business days.",
}


# --- /health et authentification ------------------------------------------------
def test_health_is_public_and_reports_state(secured, client):
    assert secured.get("/health").json() == {"status": "ok", "llm": False, "auth": True}
    assert client.get("/health").json()["auth"] is False


@pytest.mark.parametrize(
    "method, path",
    [("post", "/analyze"), ("post", "/escalations"), ("post", "/approved-replies")],
)
def test_routes_need_the_key_when_one_is_configured(secured, method, path):
    assert getattr(secured, method)(path, json={}).status_code == 401
    wrong = getattr(secured, method)(path, json={}, headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401


def test_correct_key_is_accepted(secured):
    response = secured.post(
        "/analyze", json={"message": "How long does delivery take?"}, headers={"X-API-Key": KEY}
    )
    assert response.status_code == 200


def test_without_configured_key_any_header_is_accepted(client):
    placeholder = {"X-API-Key": "change-me-if-SUPPORT_API_KEY-is-set"}
    assert client.post("/analyze", json={"message": "hi"}, headers=placeholder).status_code == 200
    assert client.post("/analyze", json={"message": "hi"}).status_code == 200


# --- /analyze ------------------------------------------------------------------------
def test_ordinary_message_goes_to_human_approval(client):
    data = client.post("/analyze", json={"message": "How long does standard delivery take?"}).json()
    assert data["route"] == "await_human_approval"
    assert data["needs_human"] is False and data["handoff"] is None
    assert data["handoff_text"] is None
    assert "3 to 5 business days" in data["suggested_response"]


def test_sensitive_message_is_escalated_with_a_summary(client):
    data = client.post(
        "/analyze", json={"message": "Someone hacked my account, order ORD-10001"}
    ).json()
    assert data["route"] == "escalate_to_human" and data["priority"] == "urgent"
    assert data["handoff"]["order_number"] == "ORD-10001"
    assert "Recommended next steps:" in data["handoff_text"]


def test_history_is_used(client):
    data = client.post(
        "/analyze",
        json={
            "message": "Still nothing and this is terrible",
            "history": ["My order is late, this is unacceptable"],
        },
    ).json()
    assert data["route"] == "escalate_to_human"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 5001},
        {"message": "hi", "history": ["a"] * 21},
        {"message": "hi", "history": "not a list"},
    ],
)
def test_analyze_validates_input(client, body):
    assert client.post("/analyze", json=body).status_code == 422


# --- /escalations ------------------------------------------------------------------------
def test_escalation_is_logged_without_sensitive_data(client, runtime):
    message = "Someone hacked my account, card 4111 1111 1111 1111, mail jean@mail.com"
    analysis = client.post("/analyze", json={"message": message}).json()
    response = client.post(
        "/escalations", json={"customer_message": message, "handoff": analysis["handoff"]}
    )
    assert response.status_code == 200 and response.json() == {"status": "recorded"}
    text = (runtime / "escalations.jsonl").read_text(encoding="utf-8")
    assert "4111" not in text and "jean@mail.com" not in text
    entry = _lines(runtime / "escalations.jsonl")[0]
    assert entry["handoff"]["priority"] == "urgent" and entry["recorded_at"]


def test_escalation_redacts_even_forged_handoff_text(client, runtime):
    analysis = client.post("/analyze", json={"message": "I want a human"}).json()
    handoff = analysis["handoff"]
    handoff["conversation_summary"] = "Card 4111 1111 1111 1111"
    client.post("/escalations", json={"customer_message": "x", "handoff": handoff})
    assert "4111" not in (runtime / "escalations.jsonl").read_text(encoding="utf-8")


def test_escalation_requires_a_valid_handoff(client):
    assert client.post("/escalations", json={"customer_message": "x"}).status_code == 422
    assert client.post("/escalations", json={"customer_message": "x", "handoff": {}}).status_code == 422


# --- /approved-replies ----------------------------------------------------------------------
def test_approved_reply_is_only_simulated(client, runtime):
    response = client.post("/approved-replies", json=APPROVED)
    assert response.status_code == 200
    assert response.json()["sent"] is False and response.json()["simulated"] is True
    entry = _lines(runtime / "outbox_simulation.jsonl")[0]
    assert entry["sent"] is False and entry["approved_by"] == "Amina"


def test_approved_reply_is_redacted_and_flags_promises(client, runtime):
    body = {**APPROVED, "reply": "Your refund has been approved. Write to jean@mail.com"}
    response = client.post("/approved-replies", json=body)
    assert response.status_code == 200 and response.json()["warnings"]
    saved = (runtime / "outbox_simulation.jsonl").read_text(encoding="utf-8")
    assert "jean@mail.com" not in saved


@pytest.mark.parametrize(
    "change",
    [
        {"approved": False},
        {"approved": None},
        {"reply": ""},
        {"reply": "   "},
        {"approved_by": ""},
        {"approved_by": "  "},
        {"reply": "x" * 3001},
    ],
)
def test_nothing_is_recorded_without_a_clear_human_approval(client, runtime, change):
    assert client.post("/approved-replies", json={**APPROVED, **change}).status_code == 422
    assert not (runtime / "outbox_simulation.jsonl").exists()


def test_missing_field_is_refused(client):
    body = {k: v for k, v in APPROVED.items() if k != "approved"}
    assert client.post("/approved-replies", json=body).status_code == 422
