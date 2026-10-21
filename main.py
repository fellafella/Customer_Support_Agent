"""Point d'entrée CLI de la Phase 1 : analyse un message client fictif.

Usage :
    python main.py "Where is my order ORD-12345?"
    python main.py            # lance quelques exemples
"""

from __future__ import annotations

import sys

from src.analyzer import analyze_message

SAMPLE_MESSAGES = [
    "Hi, how long does standard delivery take?",
    "Where is my order ORD-48213? It is late and I'm really frustrated!",
    "Bonjour, je veux un remboursement pour ma commande 55120, le produit est cassé.",
    "Someone made an unauthorized payment with my card, this is urgent!",
    "I want to talk to a human please.",
]


def main() -> None:
    """Affiche l'analyse JSON du message passé en argument (ou d'exemples)."""
    messages = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else SAMPLE_MESSAGES
    for text in messages:
        result = analyze_message(text)
        print(f"\n> {text}")
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
