"""Réponses de secours et réponses ancrées dans la base de connaissances.

La génération par LLM arrivera en Phase 3. D'ici là, on produit des réponses
sûres : soit un message générique, soit une citation fidèle de la fiche
retrouvée. Aucune promesse n'est faite sur un remboursement, une livraison ou
une politique qui ne figure pas dans la fiche.
"""

from __future__ import annotations

from src.models import RetrievedInformation

_ESCALATION_TEXT = {
    "en": (
        "Thank you for contacting us. Your request needs to be reviewed by a "
        "member of our support team, who will get back to you shortly."
    ),
    "fr": (
        "Merci de nous avoir contactés. Votre demande doit être examinée par un "
        "membre de notre équipe support, qui reviendra vers vous rapidement."
    ),
}

_DEFAULT_TEXT = {
    "en": (
        "Thank you for your message. Could you give us a few more details "
        "so we can help you?"
    ),
    "fr": (
        "Merci pour votre message. Pourriez-vous nous donner quelques détails "
        "supplémentaires afin que nous puissions vous aider ?"
    ),
}

_GROUNDED_INTRO = {
    "en": "Thank you for your message. Here is the information we have:",
    "fr": (
        "Merci pour votre message. Voici l'information dont nous disposons "
        "(texte original en anglais) :"
    ),
}

_GROUNDED_OUTRO = {
    "en": "If this does not answer your question, please let us know.",
    "fr": "Si cela ne répond pas à votre question, n'hésitez pas à nous le dire.",
}


def build_fallback_response(language: str, needs_human: bool) -> str:
    """Retourne une réponse sûre et générique dans la langue du client."""
    texts = _ESCALATION_TEXT if needs_human else _DEFAULT_TEXT
    return texts.get(language, texts["en"])


def build_grounded_response(language: str, info: RetrievedInformation) -> str:
    """Construit une réponse qui cite uniquement la fiche retrouvée.

    Le texte de la fiche est repris tel quel : rien n'est ajouté ni inventé.
    """
    lang = language if language in _GROUNDED_INTRO else "en"
    return (
        f"{_GROUNDED_INTRO[lang]}\n\n"
        f"{info.title}: {info.content}\n\n"
        f"{_GROUNDED_OUTRO[lang]}"
    )
