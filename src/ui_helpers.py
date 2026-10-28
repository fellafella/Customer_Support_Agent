"""Fonctions utilitaires de l'interface (testables sans Streamlit)."""

from __future__ import annotations

from src.keywords import FORBIDDEN_CLAIM_KEYWORDS, find_matches, normalize_text
from src.models import Priority

# Couleurs de la syntaxe Streamlit ":couleur[texte]".
PRIORITY_COLORS: dict[Priority, str] = {
    Priority.LOW: "green",
    Priority.MEDIUM: "blue",
    Priority.HIGH: "orange",
    Priority.URGENT: "red",
}

SAMPLE_MESSAGES: tuple[str, ...] = (
    "Hi, how long does standard delivery take?",
    "Quel est le délai de livraison ?",
    "Where is my order ORD-48213? It is late and I'm really frustrated!",
    "Bonjour, je veux un remboursement pour ma commande 55120, le produit est cassé.",
    "Someone made an unauthorized payment with my card, this is urgent!",
    "I want to talk to a human please.",
    "Do you ship to Mars?",
)

MAX_MESSAGE_LENGTH = 5000


def priority_markdown(priority: Priority) -> str:
    """Retourne la priorité en gras et en couleur (ex. ':red[**URGENT**]')."""
    return f":{PRIORITY_COLORS[priority]}[**{priority.value.upper()}**]"


def draft_warnings(text: str) -> list[str]:
    """Signale les problèmes d'un brouillon modifié à la main.

    L'humain a le dernier mot, mais on l'avertit s'il écrit une promesse que
    l'entreprise n'a pas validée (remboursement, annulation, garantie...).
    """
    warnings: list[str] = []
    if not text.strip():
        warnings.append("The draft is empty.")
    if find_matches(normalize_text(text), FORBIDDEN_CLAIM_KEYWORDS):
        warnings.append(
            "The draft seems to promise or confirm an action (refund, cancellation, "
            "guarantee...). Only keep it if a trusted system has confirmed it."
        )
    return warnings


# --- Habillage HTML de l'interface ---------------------------------------------
# Tout texte inséré dans le HTML est échappé : même les valeurs produites par
# nos propres règles passent par html.escape (défense en profondeur).

from html import escape as _escape  # noqa: E402

from src.models import AnalysisResult  # noqa: E402


def _e(value: object) -> str:
    return _escape(str(value), quote=True)


def _sentence(label: str) -> str:
    """'report_late_or_lost' -> 'Report late or lost'."""
    text = label.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def masthead_html() -> str:
    """En-tête de la page."""
    return (
        '<header class="masthead">'
        "<h1>Customer support desk</h1>"
        "<p>Paste a customer message. The agent sorts it, looks for the answer in the "
        "knowledge base and drafts a reply. You review, edit and approve it. "
        "Nothing is ever sent.</p>"
        "</header>"
    )


def empty_state_html() -> str:
    """Message affiché tant qu'aucun message n'a été analysé."""
    return (
        '<div class="empty"><strong>No case yet.</strong> '
        "Load an example or type a customer message above, then choose "
        "<strong>Analyze message</strong>.</div>"
    )


def _trail_steps(result: AnalysisResult) -> list[tuple[str, str]]:
    found = len(result.retrieved_information)
    triggers = len(result.handoff.triggers) if result.handoff else 0
    if result.needs_human:
        rules = f"{max(triggers, 1)} rule{'s' if triggers > 1 else ''} triggered"
        reply = "holding message only"
    else:
        rules = "no rule triggered"
        reply = "written by the LLM" if result.response_source == "llm" else "from a safe template"
    return [
        ("Understood", _sentence(result.intent).lower()),
        (
            "Searched the knowledge base",
            f"{found} entr{'y' if found == 1 else 'ies'} found" if found else "nothing reliable found",
        ),
        ("Checked the escalation rules", rules),
        ("Drafted a reply", reply),
    ]


def ticket_html(result: AnalysisResult, case_number: int) -> str:
    """Fiche « ticket » : titre, tampon, faits, raison et parcours de décision."""
    prio = result.priority.value
    stamp_class, stamp_text = (
        ("is-alert", "Needs a human") if result.needs_human else ("is-ready", "Draft ready")
    )
    if result.needs_human:
        reason = f'<p class="reason"><strong>Why:</strong> {_e(result.escalation_reason)}</p>'
    else:
        reason = (
            '<p class="reason is-fine">No rule asks for a human. '
            "The draft still needs your approval.</p>"
        )
    steps = "".join(
        f'<li class="step" style="--i:{i}"><span class="tick"></span>'
        f'<b>{_e(title)}</b><span class="detail">{_e(detail)}</span></li>'
        for i, (title, detail) in enumerate(_trail_steps(result))
    )
    facts = (
        ("Priority", f'<i class="dot"></i>{_e(_sentence(prio))}'),
        ("Category", _e(_sentence(result.category.value))),
        ("Sentiment", _e(_sentence(result.sentiment.value))),
        ("Order number", _e(result.order_number or "Not provided")),
    )
    facts_html = "".join(f"<div class='fact'><dt>{_e(k)}</dt><dd>{v}</dd></div>" for k, v in facts)
    return (
        f'<article class="ticket prio-{_e(prio)}">'
        '<div class="ticket-head"><div>'
        f'<p class="ticket-no">Case {int(case_number):03d}</p>'
        f'<h2 class="ticket-title">{_e(_sentence(result.intent))}</h2></div>'
        f'<div class="stamp {stamp_class}">{stamp_text}</div></div>'
        '<div class="perf"></div>'
        f'<dl class="facts">{facts_html}</dl>'
        f"{reason}"
        '<div class="perf"></div>'
        f'<ol class="trail">{steps}</ol>'
        "</article>"
    )


def relevance_bar_html(score: float) -> str:
    """Barre de pertinence d'une fiche (score entre 0 et 1)."""
    percent = max(0, min(100, round(float(score) * 100)))
    return (
        f'<div class="bar" role="img" aria-label="Relevance {percent} percent">'
        f'<i style="--w:{percent}%"></i></div>'
    )

