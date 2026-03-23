"""Tests pour le moteur RAG."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch
from src.rag_engine import RAGEngine, create_engine


class TestRAGEngineCreation:
    """Tests de la création du moteur RAG."""

    def test_create_engine_returns_rag_engine(self):
        """Vérifie que create_engine retourne une instance de RAGEngine."""
        engine = create_engine()
        assert isinstance(engine, RAGEngine)

    def test_engine_has_required_attributes(self):
        """Vérifie que le moteur a les attributs requis."""
        engine = create_engine()
        assert hasattr(engine, "vector_store")
        assert hasattr(engine, "llm")

    def test_engine_initial_state(self):
        """Vérifie l'état initial du moteur."""
        engine = RAGEngine()
        assert engine.vector_store is None
        assert engine.llm is None


class TestRAGEngineAsk:
    """Tests de la méthode ask."""

    @patch('src.rag_engine.load_vector_store')
    def test_ask_raises_if_not_initialized(self, mock_store):
        """Vérifie qu'une exception est levée si le moteur n'est pas initialisé."""
        engine = RAGEngine()

        with pytest.raises(RuntimeError, match="n'est pas initialisé"):
            engine.ask("Question?")

    def test_ask_without_vector_store(self):
        """Vérifie le comportement sans vector_store."""
        engine = RAGEngine()
        engine.llm = Mock()

        with pytest.raises(RuntimeError, match="n'est pas initialisé"):
            engine.ask("Question?")


class TestRAGEngineSearchOnly:
    """Tests de la méthode search_only."""

    @patch('src.rag_engine.load_vector_store')
    def test_search_only_raises_if_not_initialized(self, mock_store):
        """Vérifie qu'une exception est levée si le moteur n'est pas initialisé."""
        engine = RAGEngine()

        with pytest.raises(RuntimeError, match="n'est pas initialisé"):
            engine.search_only("Query")

    def test_search_only_without_vector_store(self):
        """Vérifie le comportement sans vector_store."""
        engine = RAGEngine()

        with pytest.raises(RuntimeError, match="n'est pas initialisé"):
            engine.search_only("Query")


class TestRAGEngineHealthCheck:
    """Tests de la méthode health_check."""

    def test_health_check_without_initialization(self):
        """Vérifie le health check sans initialisation."""
        engine = RAGEngine()
        result = engine.health_check()

        assert isinstance(result, dict)
        assert "status" in result
        assert "documents_count" in result


class TestRAGEnginePredefined:
    """Tests pour les questions prédéfinies."""

    def test_answer_all_predefined_raises_if_not_initialized(self):
        """Vérifie que answer_all_predefined lève une exception si non initialisé."""
        engine = RAGEngine()

        with pytest.raises(RuntimeError, match="n'est pas initialisé"):
            engine.answer_all_predefined()


class TestInitialize:
    """Tests de la méthode initialize."""

    @patch('src.rag_engine.ChatOllama')
    @patch('src.rag_engine.load_vector_store')
    def test_initialize_creates_llm(self, mock_store, mock_ollama):
        """Vérifie que initialize crée le LLM."""
        mock_store_instance = Mock()
        mock_store_instance.count.return_value = 10
        mock_store.return_value = mock_store_instance

        engine = RAGEngine()
        engine.initialize()

        assert engine.llm is not None
        mock_ollama.assert_called_once()

    @patch('src.rag_engine.ChatOllama')
    @patch('src.rag_engine.load_vector_store')
    def test_initialize_loads_vector_store(self, mock_store, mock_ollama):
        """Vérifie que initialize charge le vector_store."""
        mock_store_instance = Mock()
        mock_store_instance.count.return_value = 10
        mock_store.return_value = mock_store_instance

        engine = RAGEngine()
        engine.initialize()

        assert engine.vector_store is not None
        mock_store.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])