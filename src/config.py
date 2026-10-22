"""Configuration de l'application (clé API, modèle) lue depuis l'environnement.

La clé API vit dans le fichier ``.env`` (exclu de Git) ou dans une variable
d'environnement. Elle n'est jamais affichée : ``Settings`` masque ce champ.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_SECONDS = 20.0
_PLACEHOLDER_KEYS = {"", "your-api-key-here", "changeme"}


@dataclass(frozen=True)
class Settings:
    """Paramètres du LLM."""

    provider: str = "gemini"
    api_key: str | None = field(default=None, repr=False)  # repr=False : jamais affichée
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def is_configured(self) -> bool:
        """Vrai si une vraie clé API est présente (pas le texte d'exemple)."""
        return bool(self.api_key) and self.api_key.strip().lower() not in _PLACEHOLDER_KEYS


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Lit la configuration.

    Args:
        env: dictionnaire à utiliser à la place de l'environnement (tests).
            Si None, charge d'abord le fichier ``.env`` du projet.
    """
    if env is None:
        load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ

    api_key = (env.get("LLM_API_KEY") or env.get("GEMINI_API_KEY") or "").strip() or None
    try:
        timeout = float(env.get("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        if timeout <= 0:
            raise ValueError
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SECONDS
    return Settings(
        provider=(env.get("LLM_PROVIDER") or "gemini").strip().lower(),
        api_key=api_key,
        model=(env.get("LLM_MODEL") or DEFAULT_MODEL).strip(),
        timeout_seconds=timeout,
    )
