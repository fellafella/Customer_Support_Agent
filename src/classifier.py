"""Classification basée sur des règles : intention, catégorie, priorité, sentiment.

Phase 1 : approche volontairement simple (mots-clés, FR/EN). Elle sera
complétée par un LLM en Phase 3, mais ces règles resteront utiles comme
filet de sécurité et comme référence pour les tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.keywords import (
    ENGLISH_MARKERS,
    FRENCH_MARKERS,
    FRAUD_SECURITY_KEYWORDS,
    HUMAN_REQUEST_KEYWORDS,
    NEGATION_PATTERN,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    URGENCY_KEYWORDS,
    find_matches,
    normalize_text,
)
from src.models import Category, Priority, Sentiment


@dataclass(frozen=True)
class IntentRule:
    """Une règle : si ces mots-clés apparaissent, l'intention est probable."""

    intent: str
    category: Category
    keywords: tuple[str, ...]


# L'ordre compte : en cas d'égalité de score, la règle la plus haute gagne.
INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule("report_fraud_or_security_issue", Category.OTHER, FRAUD_SECURITY_KEYWORDS),
    IntentRule(
        "report_damaged_item", Category.REFUNDS_RETURNS,
        ("damaged", "broken", "defective", "faulty", "endommage", "casse",
         "defectueux", "abime"),
    ),
    IntentRule(
        "request_refund", Category.REFUNDS_RETURNS,
        ("refund", "money back", "remboursement", "rembourse", "reimbursement"),
    ),
    IntentRule(
        "return_item", Category.REFUNDS_RETURNS,
        ("return", "send back", "retour", "renvoyer"),
    ),
    IntentRule(
        "cancel_or_modify_order", Category.ORDERS_DELIVERY,
        ("cancel", "annuler", "change my address", "change the address",
         "modifier ma commande", "changer mon adresse", "modifier mon adresse"),
    ),
    IntentRule(
        "report_late_or_lost_delivery", Category.ORDERS_DELIVERY,
        ("late!", "lost", "missing", "not arrived", "never arrived", "hasnt arrived",
         "retard", "pas recu", "perdu", "jamais recu", "toujours pas"),
    ),
    IntentRule(
        "track_order", Category.ORDERS_DELIVERY,
        ("track", "where is my order", "order status", "status of my order",
         "suivi", "suivre", "ou est ma commande", "ou est mon colis"),
    ),
    IntentRule(
        "delivery_information", Category.ORDERS_DELIVERY,
        ("delivery", "shipping", "ship!", "livraison", "expedition", "delai", "colis"),
    ),
    IntentRule(
        "login_problem", Category.TECHNICAL_SUPPORT,
        ("login", "log in", "password", "sign in", "mot de passe", "connexion",
         "me connecter"),
    ),
    IntentRule(
        "device_problem", Category.TECHNICAL_SUPPORT,
        ("not turning on", "wont turn on", "does not turn on", "doesnt turn on",
         "wont charge", "not charging", "ne sallume pas", "ne charge pas"),
    ),
    IntentRule(
        "technical_error", Category.TECHNICAL_SUPPORT,
        ("error", "bug", "crash", "not working", "doesnt work", "erreur",
         "ne fonctionne pas", "ne marche pas", "plante"),
    ),
    IntentRule(
        "file_complaint", Category.COMPLAINT,
        ("complaint", "unacceptable", "terrible", "awful", "worst", "plainte",
         "reclamation", "inacceptable", "honteux", "scandale"),
    ),
    IntentRule(
        "product_question", Category.PRODUCT_INFO,
        ("novabuds", "warranty", "garantie", "price", "prix", "colour", "color",
         "couleur", "battery life", "how much", "features", "specification",
         "payment", "paiement", "paypal", "credit card", "carte bancaire",
         "product", "produit", "ecouteurs", "earbuds"),
    ),
)

FALLBACK_INTENT = "general_inquiry"
HUMAN_REQUEST_INTENT = "request_human_agent"

# Format fictif du projet : "ORD-12345".
_ORDER_PREFIXED = re.compile(r"\b(?:ORD|CMD)[-\s]?(\d{4,10})\b", re.IGNORECASE)
_ORDER_AFTER_WORD = re.compile(
    r"\b(?:order|commande)\s*(?:number|no\.?|num(?:[eé]ro)?|n°|#)?\s*[:#]?\s*(\d{4,10})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Classification:
    """Résultat de la classification d'un message."""

    intent: str
    category: Category
    priority: Priority
    sentiment: Sentiment
    order_number: str | None
    language: str
    matched_keywords: tuple[str, ...]


def detect_intent(normalized: str) -> tuple[str, Category, tuple[str, ...]]:
    """Choisit l'intention dont le plus de mots-clés apparaissent dans le message."""
    best_rule: IntentRule | None = None
    best_matches: list[str] = []
    for rule in INTENT_RULES:
        matches = find_matches(normalized, rule.keywords)
        if len(matches) > len(best_matches):  # '>' : la 1re règle gagne les égalités
            best_rule, best_matches = rule, matches
    if best_rule is not None:
        return best_rule.intent, best_rule.category, tuple(best_matches)
    human = find_matches(normalized, HUMAN_REQUEST_KEYWORDS)
    if human:
        return HUMAN_REQUEST_INTENT, Category.OTHER, tuple(human)
    return FALLBACK_INTENT, Category.OTHER, ()


def detect_sentiment(normalized: str) -> Sentiment:
    """Estime le sentiment en comparant mots négatifs et positifs."""
    negative = len(find_matches(normalized, NEGATIVE_KEYWORDS))
    without_negations = NEGATION_PATTERN.sub(" ", normalized)
    positive = len(find_matches(without_negations, POSITIVE_KEYWORDS))
    if negative > positive:
        return Sentiment.NEGATIVE
    if positive > negative:
        return Sentiment.POSITIVE
    return Sentiment.NEUTRAL


def detect_priority(normalized: str, category: Category, sentiment: Sentiment) -> Priority:
    """Détermine la priorité à partir de la sécurité, de l'urgence et du ton."""
    if find_matches(normalized, FRAUD_SECURITY_KEYWORDS):
        return Priority.URGENT
    urgent_words = bool(find_matches(normalized, URGENCY_KEYWORDS))
    if urgent_words and sentiment == Sentiment.NEGATIVE:
        return Priority.URGENT
    if urgent_words or sentiment == Sentiment.NEGATIVE or category == Category.COMPLAINT:
        return Priority.HIGH
    if category in (
        Category.ORDERS_DELIVERY,
        Category.REFUNDS_RETURNS,
        Category.TECHNICAL_SUPPORT,
    ):
        return Priority.MEDIUM
    return Priority.LOW


def extract_order_number(text: str) -> str | None:
    """Extrait un numéro de commande (format fictif) et le met sous la forme ORD-12345."""
    match = _ORDER_PREFIXED.search(text) or _ORDER_AFTER_WORD.search(text)
    return f"ORD-{match.group(1)}" if match else None


def detect_language(normalized: str) -> str:
    """Devine la langue ('fr' ou 'en') ; l'anglais est la langue par défaut."""
    words = re.findall(r"[a-z]+", normalized)
    french = sum(1 for w in words if w in FRENCH_MARKERS)
    english = sum(1 for w in words if w in ENGLISH_MARKERS)
    return "fr" if french > english else "en"


def classify_message(text: str) -> Classification:
    """Analyse un message client et retourne sa classification."""
    normalized = normalize_text(text)
    intent, category, matched = detect_intent(normalized)
    sentiment = detect_sentiment(normalized)
    return Classification(
        intent=intent,
        category=category,
        priority=detect_priority(normalized, category, sentiment),
        sentiment=sentiment,
        order_number=extract_order_number(text),
        language=detect_language(normalized),
        matched_keywords=matched,
    )
