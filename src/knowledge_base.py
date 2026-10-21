"""Chargement et validation de la base de connaissances locale (JSON).

La recherche (retrieval) sera ajoutée en Phase 2. Ici, on s'assure seulement
que les données sont chargées et valides. Les entrées sont des objets simples,
ce qui permettra plus tard de les indexer dans une base vectorielle.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from src.models import Category

DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


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