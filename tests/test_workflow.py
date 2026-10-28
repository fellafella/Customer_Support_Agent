"""Tests du workflow n8n : structure du graphe, garde-fous et contrat avec l'API.

On ne peut pas exécuter n8n ici : on vérifie donc le fichier d'import lui-même
(graphe, approbation obligatoire, absence de secrets) et que les appels que le
workflow fait correspondent exactement à ce que l'API accepte.
"""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.knowledge_base import load_knowledge_base

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / "n8n" / "support_agent_workflow.json"
WF = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
NODES = {n["name"]: n for n in WF["nodes"]}

TRIGGER = "Receive request"
GATE_WAIT = "Wait for human approval"
GATE_IF = "Approved?"
SEND = "Save to simulated outbox"


def _edges(skip: set[str] = frozenset()) -> dict[str, list[tuple[int, str]]]:
    """Arêtes du graphe : nœud -> [(sortie, nœud suivant)], en ignorant certains nœuds."""
    result: dict[str, list[tuple[int, str]]] = {}
    for source, outputs in WF["connections"].items():
        if source in skip:
            continue
        for index, targets in enumerate(outputs["main"]):
            for target in targets:
                if target["node"] not in skip:
                    result.setdefault(source, []).append((index, target["node"]))
    return result


def _reachable(start: str, skip: set[str] = frozenset(), first_output: int | None = None) -> set[str]:
    edges = _edges(skip)
    seen: set[str] = set()
    stack = [
        target
        for index, target in edges.get(start, [])
        if first_output is None or index == first_output
    ]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(target for _, target in edges.get(node, []))
    return seen


# --- Structure -------------------------------------------------------------------
def test_file_is_valid_and_importable_shape():
    assert {"name", "nodes", "connections"} <= WF.keys()
    assert WF["active"] is False  # jamais activé par défaut
    assert len(NODES) == len(WF["nodes"]), "noms de nœuds en double"
    ids = [n["id"] for n in WF["nodes"]]
    assert len(set(ids)) == len(ids)


def test_connections_reference_existing_nodes():
    for source, outputs in WF["connections"].items():
        assert source in NODES
        for targets in outputs["main"]:
            for target in targets:
                assert target["node"] in NODES


def test_single_trigger_and_everything_is_reachable():
    triggers = [n for n in WF["nodes"] if n["type"].endswith(".webhook")]
    assert len(triggers) == 1 and triggers[0]["name"] == TRIGGER
    assert triggers[0]["parameters"]["httpMethod"] == "POST"
    assert _reachable(TRIGGER) | {TRIGGER} == set(NODES)


# --- L'approbation humaine est obligatoire ------------------------------------------
def test_no_path_to_sending_without_the_human_gate():
    """Si on retire l'étape d'approbation, le nœud d'envoi devient inatteignable."""
    assert SEND in _reachable(TRIGGER)
    assert SEND not in _reachable(TRIGGER, skip={GATE_WAIT})
    assert SEND not in _reachable(TRIGGER, skip={GATE_IF})


def test_the_gate_is_a_form_that_a_person_must_fill():
    wait = NODES[GATE_WAIT]
    assert wait["type"] == "n8n-nodes-base.wait"
    assert wait["parameters"]["resume"] == "form"
    fields = {f["fieldLabel"]: f for f in wait["parameters"]["formFields"]["values"]}
    assert fields["Decision"]["requiredField"] is True
    assert fields["Your name"]["requiredField"] is True
    options = [o["option"] for o in fields["Decision"]["fieldOptions"]["values"]]
    assert "Reject" in options


def test_rejection_and_escalations_never_reach_the_outbox():
    assert SEND not in _reachable("Needs a human?", first_output=0)  # branche « humain »
    assert SEND not in _reachable(GATE_IF, first_output=1)  # branche « rejeté »
    assert SEND not in _reachable("Analyze message", first_output=1)  # branche « erreur »


def test_sensitive_cases_skip_the_ai_draft_review_path():
    escalate = _reachable("Needs a human?", first_output=0)
    assert GATE_WAIT not in escalate and "Prepare review" not in escalate
    assert "Record escalation" in escalate


def test_api_failure_hands_over_to_a_human():
    analyze = NODES["Analyze message"]
    assert analyze["onError"] == "continueErrorOutput"
    assert analyze["retryOnFail"] is True
    outputs = WF["connections"]["Analyze message"]["main"]
    assert outputs[1][0]["node"].startswith("Hand over to a human")


def test_routing_uses_the_api_route_field():
    condition = NODES["Needs a human?"]["parameters"]["conditions"]["conditions"][0]
    assert condition["leftValue"] == "={{ $json.route }}"
    assert condition["rightValue"] == "escalate_to_human"


# --- Sécurité ------------------------------------------------------------------------
def test_no_credentials_or_secrets_in_the_file():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert '"credentials"' not in text
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{20,}", text)


def test_every_http_call_targets_the_local_api():
    urls = [
        n["parameters"]["url"] for n in WF["nodes"] if n["type"].endswith(".httpRequest")
    ]
    assert urls and all(u.startswith("http://127.0.0.1:8000/") for u in urls)


def test_expressions_only_reference_existing_nodes():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    for name in re.findall(r"\$\('([^']+)'\)", text):
        assert name in NODES, name


# --- Contrat avec l'API : ce que le workflow envoie est accepté ----------------------
@pytest.fixture
def client(tmp_path):
    app = create_app(
        kb=load_knowledge_base(), llm=None, runtime_dir=tmp_path / "rt", api_key=None
    )
    return TestClient(app)


def test_escalated_flow_contract(client, tmp_path):
    message = "Someone hacked my account, card 4111 1111 1111 1111"
    analysis = client.post("/analyze", json={"message": message, "history": []}).json()
    assert analysis["route"] == "escalate_to_human"
    # Le workflow transmet `handoff` tel quel à /escalations.
    logged = client.post(
        "/escalations", json={"customer_message": message, "handoff": analysis["handoff"]}
    )
    assert logged.status_code == 200
    saved = (tmp_path / "rt" / "escalations.jsonl").read_text(encoding="utf-8")
    assert "4111" not in saved


def test_approval_flow_contract(client, tmp_path):
    message = "How long does standard delivery take?"
    analysis = client.post("/analyze", json={"message": message}).json()
    assert analysis["route"] == "await_human_approval"
    # Corps exact construit par le nœud « Save to simulated outbox » (réponse telle quelle).
    body = {
        "approved": True,
        "approved_by": "Amina",
        "customer_message": message,
        "reply": analysis["suggested_response"],
    }
    response = client.post("/approved-replies", json=body)
    assert response.json()["sent"] is False
    # Variante « réponse modifiée » avec champ vide : l'API refuse plutôt que d'inventer.
    assert client.post("/approved-replies", json={**body, "reply": ""}).status_code == 422
