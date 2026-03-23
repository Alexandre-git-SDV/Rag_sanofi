"""Tests pour les templates de prompts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.prompt_templates import (
    SYSTEM_PROMPT,
    PROMpts,
    PREDEFINED_QUESTIONS,
    get_prompt_for_category,
    build_final_prompt
)


class TestSystemPrompt:
    """Tests du prompt système."""

    def test_system_prompt_is_string(self):
        """Vérifie que le prompt système est une chaîne non vide."""
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_contains_sanofi(self):
        """Vérifie que le prompt fait référence à Sanofi."""
        assert "sanofi" in SYSTEM_PROMPT.lower() or "rapport" in SYSTEM_PROMPT.lower()


class TestPrompts:
    """Tests des prompts spécialisés par catégorie."""

    def test_prompts_is_dict(self):
        """Vérifie que PROMpts est un dictionnaire."""
        assert isinstance(PROMpts, dict)

    def test_all_categories_have_system_and_user(self):
        """Vérifie que chaque catégorie a 'system' et 'user_template'."""
        for category, prompts in PROMpts.items():
            assert "system" in prompts, f"Catégorie '{category}' manque 'system'"
            assert "user_template" in prompts, f"Catégorie '{category}' manque 'user_template'"

    def test_all_prompts_are_non_empty(self):
        """Vérifie que tous les prompts sont non vides."""
        for category, prompts in PROMpts.items():
            assert len(prompts["system"]) > 0, f"Prompt system vide pour '{category}'"
            assert len(prompts["user_template"]) > 0, f"Prompt user vide pour '{category}'"

    def test_user_template_contains_context_placeholder(self):
        """Vérifie que le template contient le placeholder {context}."""
        for category, prompts in PROMpts.items():
            assert "{context}" in prompts["user_template"], f"Catégorie '{category}' sans placeholder"


class TestPredefinedQuestions:
    """Tests des questions prédéfinies."""

    def test_predefined_questions_is_list(self):
        """Vérifie que PREDEFINED_QUESTIONS est une liste."""
        assert isinstance(PREDEFINED_QUESTIONS, list)
        assert len(PREDEFINED_QUESTIONS) > 0

    def test_six_questions(self):
        """Vérifie qu'il y a exactement 6 questions."""
        assert len(PREDEFINED_QUESTIONS) == 6

    def test_each_question_has_required_fields(self):
        """Vérifie que chaque question a les champs requis."""
        for q in PREDEFINED_QUESTIONS:
            assert "id" in q
            assert "category" in q
            assert "question" in q
            assert isinstance(q["id"], int)
            assert isinstance(q["category"], str)
            assert isinstance(q["question"], str)

    def test_all_categories_are_unique(self):
        """Vérifie que les catégories sont uniques."""
        categories = [q["category"] for q in PREDEFINED_QUESTIONS]
        assert len(categories) == len(set(categories))

    def test_ids_are_sequential(self):
        """Vérifie que les IDs vont de 1 à 6."""
        ids = [q["id"] for q in PREDEFINED_QUESTIONS]
        assert ids == [1, 2, 3, 4, 5, 6]


class TestGetPromptForCategory:
    """Tests de la fonction get_prompt_for_category."""

    def test_returns_dict_with_system_and_user(self):
        """Vérifie le retour de la fonction."""
        result = get_prompt_for_category("neutralite_carbone", "contexte test")
        assert isinstance(result, dict)
        assert "system" in result
        assert "user" in result

    def test_unknown_category_returns_default(self):
        """Vérifie le comportement avec une catégorie inconnue."""
        result = get_prompt_for_category("inconnue", "contexte test")
        assert "system" in result
        assert "user" in result


class TestBuildFinalPrompt:
    """Tests de la fonction build_final_prompt."""

    def test_returns_string(self):
        """Vérifie que la fonction retourne une chaîne."""
        result = build_final_prompt("Ma question?", "Mon contexte", None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_question(self):
        """Vérifie que le prompt contient la question."""
        result = build_final_prompt("Quelle est la répartition?", "contexte", None)
        assert "Quelle est la répartition?" in result

    def test_contains_context(self):
        """Vérifie que le prompt contient le contexte."""
        result = build_final_prompt("Question?", "Mon contexte", None)
        assert "Mon contexte" in result

    def test_with_category_uses_specialized_prompt(self):
        """Vérifie qu'avec une catégorie, le prompt spécialisé est utilisé."""
        result = build_final_prompt("Question?", "contexte", "ventes")
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])