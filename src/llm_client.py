"""Accès au LLM (Google Gemini via Google AI Studio).

Le reste de l'application ne dépend que du protocole ``LLMClient`` : changer de
fournisseur (OpenAI, Mistral...) revient à écrire une autre classe qui
implémente ``generate_structured``. Toute erreur (réseau, quota, clé invalide,
sortie invalide) est convertie en ``LLMError`` pour que l'appelant puisse
basculer sur une réponse de secours.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from src.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Erreur du LLM ; le message ne contient jamais de données client ni de clé."""


class LLMClient(Protocol):
    """Contrat minimal d'un client LLM."""

    def generate_structured(
        self, *, system_instruction: str, prompt: str, schema: type[T]
    ) -> T:
        """Retourne une réponse validée selon ``schema`` ou lève ``LLMError``."""
        ...


class GeminiClient:
    """Client Gemini (SDK officiel ``google-genai``) avec sortie JSON structurée."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client if client is not None else self._build_client()

    def _build_client(self) -> Any:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError("Le paquet 'google-genai' n'est pas installé.") from exc
        return genai.Client(
            api_key=self._settings.api_key,
            http_options=types.HttpOptions(timeout=int(self._settings.timeout_seconds * 1000)),
        )

    def generate_structured(
        self, *, system_instruction: str, prompt: str, schema: type[T]
    ) -> T:
        try:
            from google.genai import types
        except ImportError as exc:
            raise LLMError("Le paquet 'google-genai' n'est pas installé.") from exc

        try:
            response = self._client.models.generate_content(
                model=self._settings.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except Exception as exc:  # le SDK lève des types variés (réseau, quota, 4xx...)
            # On ne garde que le nom de l'erreur : pas de contenu client dans les logs.
            raise LLMError(f"Appel au LLM échoué ({type(exc).__name__}).") from exc

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate_json(response.text)
        except Exception as exc:
            raise LLMError("Sortie du LLM invalide (JSON non conforme).") from exc


def create_llm_client(settings: Settings) -> LLMClient | None:
    """Crée le client configuré, ou None si aucune clé API valide n'est définie."""
    if not settings.is_configured:
        return None
    if settings.provider != "gemini":
        logger.warning("Fournisseur LLM non pris en charge : %s", settings.provider)
        return None
    try:
        return GeminiClient(settings)
    except LLMError as exc:
        logger.warning("LLM indisponible : %s", exc)
        return None
