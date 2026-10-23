"""Modèles de données Pydantic de l'agent de support client.

Ces modèles servent à valider tout ce qui entre et sort de l'agent :
si une valeur est invalide (ex. priorité inconnue), Pydantic lève une erreur
au lieu de laisser passer des données incohérentes.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(str, Enum):
    """Catégories de demandes clients."""

    PRODUCT_INFO = "product_information"
    ORDERS_DELIVERY = "orders_and_delivery"
    REFUNDS_RETURNS = "refunds_and_returns"
    TECHNICAL_SUPPORT = "technical_support"
    COMPLAINT = "complaint"
    OTHER = "other"


class Priority(str, Enum):
    """Niveaux de priorité (ordre croissant d'urgence)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Sentiment(str, Enum):
    """Sentiment exprimé dans le message."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class CustomerMessage(BaseModel):
    """Message brut d'un client (entrée de l'agent)."""

    text: str = Field(..., min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def _strip_and_check_not_blank(cls, value: str) -> str:
        """Supprime les espaces autour et refuse un message vide."""
        value = value.strip()
        if not value:
            raise ValueError("Le message ne peut pas être vide.")
        return value


class RetrievedInformation(BaseModel):
    """Un passage de la base de connaissances retrouvé pour une demande."""

    doc_id: str
    title: str
    content: str
    score: float = Field(..., ge=0.0, le=1.0)


class EscalationDecision(BaseModel):
    """Résultat de l'évaluation des règles d'escalade."""

    needs_human: bool
    reason: str | None = None
    triggers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reason_matches_decision(self) -> "EscalationDecision":
        if self.needs_human and not self.reason:
            raise ValueError("Une raison est obligatoire quand needs_human est vrai.")
        if not self.needs_human and (self.reason or self.triggers):
            raise ValueError("Aucune raison/déclencheur sans escalade.")
        return self


class HandoffSummary(BaseModel):
    """Résumé de cas destiné à l'agent humain qui reprend la conversation.

    Les textes du client y sont déjà expurgés des données sensibles
    (e-mails, numéros de carte, téléphones).
    """

    headline: str
    priority: Priority
    category: Category
    intent: str
    sentiment: Sentiment
    customer_language: str
    order_number: str | None = None
    escalation_reason: str
    triggers: list[str] = Field(default_factory=list)
    conversation_summary: str
    customer_messages: list[str] = Field(default_factory=list)
    relevant_kb: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    ai_actions_taken: list[str] = Field(default_factory=list)
    ai_note: str = ""


class AnalysisResult(BaseModel):
    """Analyse complète et validée d'un message client."""

    model_config = ConfigDict(use_enum_values=False)

    intent: str = Field(..., min_length=1, max_length=100)
    category: Category
    priority: Priority
    sentiment: Sentiment
    needs_human: bool
    escalation_reason: str | None = None
    retrieved_information: list[RetrievedInformation] = Field(default_factory=list)
    suggested_response: str = ""
    order_number: str | None = None
    language: str = Field(default="en", pattern=r"^(en|fr)$")
    response_source: Literal["llm", "template"] = "template"
    handoff: HandoffSummary | None = None

    @model_validator(mode="after")
    def _escalation_is_consistent(self) -> "AnalysisResult":
        """Impose une raison d'escalade si et seulement si un humain est requis."""
        if self.needs_human and not self.escalation_reason:
            raise ValueError("escalation_reason est obligatoire si needs_human est vrai.")
        if not self.needs_human and self.escalation_reason:
            raise ValueError("escalation_reason doit être vide si needs_human est faux.")
        if not self.needs_human and self.handoff is not None:
            raise ValueError("handoff doit être vide si needs_human est faux.")
        return self


class LLMReply(BaseModel):
    """Réponse structurée demandée au LLM (validée avant utilisation)."""

    answerable: bool = Field(
        ..., description="True only if the documents contain enough information to answer."
    )
    reply: str = Field(..., description="The reply to the customer, in the customer's language.")
    used_doc_ids: list[str] = Field(
        ..., description="Ids of the documents the reply is based on."
    )
