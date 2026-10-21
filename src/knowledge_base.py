"""Chargement, validation et recherche dans la base de connaissances locale (JSON).

Phase 2 : la recherche se fait par mots-clés avec un score de pertinence entre
0 et 1. Elle est volontairement simple et isolée dans ``KnowledgeBase.search``
pour pouvoir être remplacée plus tard par une recherche sémantique (embeddings
+ base vectorielle) sans toucher au reste de l'application.

Principe anti-invention : l'agent ne s'appuie QUE sur les fiches retrouvées.
Si aucune fiche n'atteint ``MIN_RELIABLE_SCORE``, l'information est considérée
comme non fiable et la demande est escaladée.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from src.keywords import find_matches, normalize_text
from src.models import Category, RetrievedInformation

DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

# Score minimal pour considérer une fiche comme une source fiable.
MIN_RELIABLE_SCORE = 0.3

# Poids d'un mot-clé : une expression de plusieurs mots est plus précise
# qu'un mot isolé.
_SINGLE_WORD_WEIGHT = 0.5
_PHRASE_WEIGHT = 1.0
# Bonus si la catégorie de la fiche correspond à celle de la demande.
_CATEGORY_BONUS = 0.25
# Total qui donne un score de 1.0.
_SCORE_DIVISOR = 1.5


class KBEntry(BaseModel):
    """Une fiche de la base de connaissances."""

    id: str = Field(..., min_length=1)
    category: Category
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    keywords: list[str] = Field(default_factory=list)


class KnowledgeBase(BaseModel):
    """Ensemble des fiches, avec contrôle d'unicité des identifiants."""

    company: str = ""
    note: str = ""
    entries: list[KBEntry]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "KnowledgeBase":
        ids = [entry.id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Les identifiants de la base de connaissances doivent être uniques.")
        return self

    def get_entry(self, entry_id: str) -> KBEntry | None:
        """Retourne la fiche correspondant à l'identifiant, ou None."""
        return next((e for e in self.entries if e.id == entry_id), None)

    def entries_by_category(self, category: Category) -> list[KBEntry]:
        """Retourne toutes les fiches d'une catégorie."""
        return [e for e in self.entries if e.category == category]

    def search(
        self,
        query: str,
        *,
        category: Category | None = None,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[RetrievedInformation]:
        """Cherche les fiches les plus pertinentes pour une question.

        Args:
            query: texte du client.
            category: catégorie détectée ; elle donne un petit bonus aux fiches
                de la même catégorie, sans jamais suffire à elle seule.
            top_k: nombre maximal de fiches retournées.
            min_score: score minimal pour être retourné.

        Returns:
            Les fiches triées par score décroissant. Liste vide si aucun mot-clé
            ne correspond : l'agent ne doit alors rien inventer.
        """
        if top_k < 1:
            return []
        normalized = normalize_text(query)
        scored: list[tuple[float, int, KBEntry]] = []
        for position, entry in enumerate(self.entries):
            matches = find_matches(normalized, tuple(entry.keywords))
            if not matches:
                continue  # sans mot-clé commun, pas de pertinence
            raw = sum(
                _PHRASE_WEIGHT if " " in normalize_text(kw.rstrip("!")) else _SINGLE_WORD_WEIGHT
                for kw in matches
            )
            if category is not None and entry.category == category:
                raw += _CATEGORY_BONUS
            score = min(1.0, raw / _SCORE_DIVISOR)
            if score >= min_score:
                scored.append((score, position, entry))
        # Tri : meilleur score d'abord ; à égalité, l'ordre du fichier JSON.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedInformation(
                doc_id=entry.id,
                title=entry.title,
                content=entry.content,
                score=round(score, 3),
            )
            for score, _, entry in scored[:top_k]
        ]


def load_knowledge_base(path: Path | str = DEFAULT_KB_PATH) -> KnowledgeBase:
    """Charge et valide la base de connaissances depuis un fichier JSON.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si le JSON est invalide ou ne respecte pas le schéma.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Base de connaissances introuvable : {file_path}")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide dans {file_path} : {exc}") from exc
    return KnowledgeBase.model_validate(raw)
