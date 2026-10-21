"""Tests de la classification (intention, catégorie, priorité, sentiment...)."""

import pytest

from src.classifier import classify_message, extract_order_number
from src.models import Category, Priority, Sentiment


@pytest.mark.parametrize(
    "text, intent, category",
    [
        ("How long does standard delivery take?", "delivery_information", Category.ORDERS_DELIVERY),
        ("Where is my order? I want to track it", "track_order", Category.ORDERS_DELIVERY),
        ("My parcel has never arrived", "report_late_or_lost_delivery", Category.ORDERS_DELIVERY),
        ("How can I return my shoes?", "return_item", Category.REFUNDS_RETURNS),
        ("What is your refund policy?", "request_refund", Category.REFUNDS_RETURNS),
        ("The item arrived damaged", "report_damaged_item", Category.REFUNDS_RETURNS),
        ("I forgot my password", "login_problem", Category.TECHNICAL_SUPPORT),
        ("My earbuds are not turning on", "device_problem", Category.TECHNICAL_SUPPORT),
        ("How long is the NovaBuds battery life?", "product_question", Category.PRODUCT_INFO),
        ("I want to file a complaint, this is unacceptable", "file_complaint", Category.COMPLAINT),
        ("Where are you located?", "general_inquiry", Category.OTHER),
        ("Je voudrais un humain", "request_human_agent", Category.OTHER),
    ],
)
def test_intent_and_category(text, intent, category):
    result = classify_message(text)
    assert result.intent == intent
    assert result.category == category


def test_french_message_is_detected():
    result = classify_message("Bonjour, où est ma commande ? Je n'ai rien reçu.")
    assert result.language == "fr"


def test_english_is_default_language():
    assert classify_message("Hello, where is my order?").language == "en"


def test_accents_do_not_matter():
    assert classify_message("Mon colis est arrivé endommagé").intent == "report_damaged_item"


def test_word_boundaries_avoid_false_positives():
    # "late" ne doit pas être trouvé dans "chocolate" ni "later".
    assert classify_message("I love chocolate").intent == "general_inquiry"
    assert classify_message("I will write later").intent == "general_inquiry"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("This is terrible, I am very angry", Sentiment.NEGATIVE),
        ("Merci, super service !", Sentiment.POSITIVE),
        ("What are your opening hours?", Sentiment.NEUTRAL),
        ("I am not happy with this", Sentiment.NEGATIVE),
    ],
)
def test_sentiment(text, expected):
    assert classify_message(text).sentiment == expected


def test_fraud_is_urgent():
    assert classify_message("There is an unauthorized charge on my card").priority == Priority.URGENT


def test_urgent_and_negative_is_urgent():
    result = classify_message("My order is late, this is unacceptable, I need it immediately")
    assert result.priority == Priority.URGENT


def test_urgent_word_alone_is_high():
    assert classify_message("Please answer urgently about delivery").priority == Priority.HIGH


def test_default_priorities():
    assert classify_message("How long does delivery take?").priority == Priority.MEDIUM
    assert classify_message("Do you have NovaBuds in white?").priority == Priority.LOW


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My order ORD-12345 is late", "ORD-12345"),
        ("ord 98765 please", "ORD-98765"),
        ("ma commande n° 55120 est cassée", "ORD-55120"),
        ("order number: 445566", "ORD-445566"),
        ("I have a question", None),
        ("My code is 12", None),
    ],
)
def test_order_number_extraction(text, expected):
    assert extract_order_number(text) == expected
