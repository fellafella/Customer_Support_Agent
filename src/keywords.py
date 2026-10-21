"""Normalisation de texte, recherche de mots-clés et listes de mots-clés.

Tous les mots-clés sont écrits SANS accent et SANS apostrophe, car le texte du
client est normalisé de la même façon avant la comparaison
("J'ai reçu" -> "jai recu").

Règle de correspondance :
- un mot-clé doit commencer au début d'un mot ("return" trouve "returned",
  mais "late" ne trouve pas "chocolate") ;
- un "!" en fin de mot-clé impose un mot entier ("late!" ne trouve pas "later").
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


def normalize_text(text: str) -> str:
    """Met le texte en minuscules, sans accents, sans apostrophes ni tirets."""
    text = text.replace("\u2019", "'").replace("`", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "")
    text = re.sub(r"[-_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1024)
def _compile(keyword: str) -> re.Pattern[str]:
    """Compile (avec cache) le motif regex d'un mot-clé."""
    normalized = normalize_text(keyword.rstrip("!"))
    whole_word = keyword.endswith("!")
    pattern = r"(?<!\w)" + re.escape(normalized) + (r"(?!\w)" if whole_word else "")
    return re.compile(pattern)


def find_matches(normalized_text: str, keywords: tuple[str, ...]) -> list[str]:
    """Retourne les mots-clés présents dans un texte DÉJÀ normalisé."""
    return [kw for kw in keywords if _compile(kw).search(normalized_text)]


# --------------------------------------------------------------------------- #
# Sécurité / fraude / vie privée -> priorité URGENT + escalade
# --------------------------------------------------------------------------- #
FRAUD_SECURITY_KEYWORDS: tuple[str, ...] = (
    "fraud", "fraudulent", "scam", "unauthorized", "unauthorised", "hacked",
    "hacker", "stolen", "identity theft", "phishing", "compromised",
    "suspicious", "data breach", "privacy", "personal data", "gdpr", "leaked",
    "fraude", "frauduleux", "arnaque", "piratage", "pirate!", "vol didentite",
    "vole", "non autorise", "donnees personnelles", "vie privee",
    "fuite de donnees", "compte pirate", "carte volee",
)

# --------------------------------------------------------------------------- #
# Demande explicite d'un humain -> escalade
# --------------------------------------------------------------------------- #
HUMAN_REQUEST_KEYWORDS: tuple[str, ...] = (
    "human", "real person", "live agent", "representative", "manager",
    "speak to someone", "talk to someone", "speak with someone",
    "talk to an agent", "speak to an agent", "supervisor",
    "un humain", "une vraie personne", "conseiller", "responsable",
    "parler a quelquun", "parler a un agent", "operateur", "superviseur",
    "agent humain",
)

# --------------------------------------------------------------------------- #
# Litiges sensibles (juridique, facturation) -> escalade
# --------------------------------------------------------------------------- #
SENSITIVE_BILLING_KEYWORDS: tuple[str, ...] = (
    "chargeback", "lawyer", "legal", "sue!", "court!", "double charged",
    "charged twice", "overcharged", "dispute", "attorney",
    "avocat", "tribunal", "action en justice", "litige", "debite deux fois",
    "deux fois debite", "double prelevement", "preleve deux fois",
)

# --------------------------------------------------------------------------- #
# Actions que l'IA ne peut pas réaliser / autoriser -> escalade
# --------------------------------------------------------------------------- #
ACTION_REQUIRED_KEYWORDS: tuple[str, ...] = (
    "cancel my order", "cancel the order", "cancel order",
    "change my address", "change the address", "update my address",
    "delete my account", "close my account",
    "refund me", "process my refund", "issue a refund", "give me a refund",
    "i want a refund", "i want my money back", "send me a replacement",
    "annuler ma commande", "annuler la commande",
    "changer mon adresse", "modifier mon adresse", "modifier ladresse",
    "supprimer mon compte", "fermer mon compte",
    "remboursez moi", "rembourse moi", "je veux un remboursement",
    "je veux etre rembourse", "je veux mon argent",
    "envoyez moi un remplacement",
)

# --------------------------------------------------------------------------- #
# Urgence exprimée
# --------------------------------------------------------------------------- #
URGENCY_KEYWORDS: tuple[str, ...] = (
    "urgent", "immediately", "asap", "right now", "emergency",
    "as soon as possible", "today!", "urgence", "immediatement",
    "tout de suite", "au plus vite", "des que possible", "en urgence",
)

# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "angry", "furious", "terrible", "awful", "worst", "unacceptable",
    "disappointed", "unhappy", "frustrated", "horrible", "ridiculous",
    "useless", "fed up", "not happy", "not satisfied", "disgusted",
    "en colere", "furieux", "inacceptable", "decu", "mecontent", "nul!",
    "honteux", "scandale", "scandaleux", "ridicule", "marre",
    "catastrophe", "pas content", "pas satisfait", "insatisfait",
)

POSITIVE_KEYWORDS: tuple[str, ...] = (
    "thank", "thanks", "great", "love", "happy", "awesome", "perfect",
    "excellent", "appreciate", "satisfied",
    "merci", "super!", "genial", "parfait", "content", "excellent",
    "ravi", "satisfait",
)

# Négation simple supprimée avant de compter les mots positifs
# ("not happy" ne doit pas compter comme positif).
NEGATION_PATTERN = re.compile(
    r"(?<!\w)(?:not|never|no|pas|jamais|ne)\s+(?:(?:very|too|vraiment|trop)\s+)?\w+"
)

# --------------------------------------------------------------------------- #
# Détection de langue (très simple : français ou anglais)
# --------------------------------------------------------------------------- #
FRENCH_MARKERS: frozenset[str] = frozenset({
    "je", "jai", "mon", "ma", "mes", "pas", "est", "une", "les", "des", "vous",
    "veux", "bonjour", "salut", "merci", "commande", "colis", "livraison",
    "remboursement", "mais", "avec", "pour", "dans", "suis", "cest", "nest",
    "ne", "quand", "comment", "pouvez", "bonsoir", "recu", "toujours",
})

ENGLISH_MARKERS: frozenset[str] = frozenset({
    "the", "my", "is", "are", "i", "you", "not", "have", "order", "where",
    "what", "how", "can", "please", "hello", "hi", "this", "was", "with",
    "for", "and", "it", "im", "ive", "when", "still", "me", "to", "do",
})
