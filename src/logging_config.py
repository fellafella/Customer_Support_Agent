"""Journaux de l'application : jamais de données client en clair.

Principe : on ne journalise que des métadonnées (intention, priorité, routes,
nombre de fiches...). En plus, un filtre masque les e-mails, numéros de carte
et téléphones de TOUT message de journal, par sécurité : si un développeur
écrit un jour un texte client dans un log, il sera expurgé.
"""

from __future__ import annotations

import logging
import os

from src.handoff import redact_sensitive

_MARKER = "_support_agent_handler"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class RedactingFilter(logging.Filter):
    """Masque les données sensibles dans le message de chaque enregistrement."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive(record.getMessage())
        record.args = ()
        return True


def configure_logging(level: str | int | None = None) -> None:
    """Configure les journaux une seule fois (appel répété sans effet).

    Le niveau vient de l'argument, sinon de la variable ``LOG_LEVEL``
    (défaut : INFO). Un niveau inconnu retombe sur INFO.
    """
    chosen = level if level is not None else os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(chosen, str):
        chosen = logging.getLevelName(chosen.strip().upper())
    if not isinstance(chosen, int):
        chosen = logging.INFO

    root = logging.getLogger()
    root.setLevel(chosen)
    if any(getattr(h, _MARKER, False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(RedactingFilter())
    setattr(handler, _MARKER, True)
    root.addHandler(handler)
