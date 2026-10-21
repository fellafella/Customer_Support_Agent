"""Réponses de secours (placeholder de la Phase 1).

La génération par LLM arrivera en Phase 3. Pour l'instant, on fournit des
réponses sûres, sans aucune promesse sur un remboursement, une livraison ou
une politique de l'entreprise.
"""

from __future__ import annotations

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


def build_fallback_response(language: str, needs_human: bool) -> str:
    """Retourne une réponse sûre et générique dans la langue du client."""
    texts = _ESCALATION_TEXT if needs_human else _DEFAULT_TEXT
    return texts.get(language, texts["en"])