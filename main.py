"""Point d'entrée CLI : analyse un message client fictif.

Usage :
    python main.py "Where is my order ORD-12345?"
    python main.py                 # lance quelques exemples
    python main.py --no-llm ...    # force le mode sans LLM (réponses de secours)

Le LLM (Gemini) est utilisé si une clé API est définie dans le fichier .env.
Les réponses ne sont JAMAIS envoyées : ce sont des brouillons pour un humain.
"""

from __future__ import annotations

import logging
import sys

from src.analyzer import analyze_message
from src.config import load_settings
from src.llm_client import create_llm_client

SAMPLE_MESSAGES = [
    "Hi, how long does standard delivery take?",
    "Quel est le délai de livraison ?",
    "Where is my order ORD-48213? It is late and I'm really frustrated!",
    "Bonjour, je veux un remboursement pour ma commande 55120, le produit est cassé.",
    "Someone made an unauthorized payment with my card, this is urgent!",
    "I want to talk to a human please.",
]


def main() -> None:
    """Affiche l'analyse JSON du message passé en argument (ou d'exemples)."""
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    args = [a for a in sys.argv[1:] if a != "--no-llm"]
    use_llm = "--no-llm" not in sys.argv[1:]

    llm = None
    if use_llm:
        settings = load_settings()
        llm = create_llm_client(settings)
        print(
            f"Mode LLM : {settings.model}"
            if llm
            else "Mode sans LLM : aucune clé API valide dans .env (réponses de secours)."
        )
    else:
        print("Mode sans LLM (--no-llm).")

    messages = [" ".join(args)] if args else SAMPLE_MESSAGES
    for text in messages:
        result = analyze_message(text, llm=llm)
        print(f"\n> {text}")
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
