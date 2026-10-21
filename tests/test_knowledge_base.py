"""Tests du chargement de la base de connaissances."""

import json

import pytest

from src.knowledge_base import load_knowledge_base
from src.models import Category


def test_default_knowledge_base_loads():
    kb = load_knowledge_base()
    assert len(kb.entries) >= 10
    assert kb.get_entry("kb-001") is not None
    assert kb.get_entry("does-not-exist") is None


def test_every_category_except_other_has_entries():
    kb = load_knowledge_base()
    for category in Category:
        assert kb.entries_by_category(category), f"Aucune fiche pour {category}"


def test_all_entries_have_keywords():
    assert all(entry.keywords for entry in load_knowledge_base().entries)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_knowledge_base("nope.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_knowledge_base(path)


def test_duplicate_ids_rejected(tmp_path):
    entry = {"id": "a", "category": "other", "title": "t", "content": "c"}
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"entries": [entry, entry]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_knowledge_base(path)


def test_invalid_category_rejected(tmp_path):
    entry = {"id": "a", "category": "banana", "title": "t", "content": "c"}
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_knowledge_base(path)
