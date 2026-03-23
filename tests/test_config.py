"""Tests pour le module de configuration."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.config import (
    BASE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    PDF_PATH,
    MAX_TOKENS_RESPONSE,
    TEMPERATURE
)


class TestConfig:
    """Tests des variables de configuration."""

    def test_base_dir_is_path(self):
        """Vérifie que BASE_DIR est un objet Path valide."""
        assert isinstance(BASE_DIR, Path)
        assert BASE_DIR.exists()

    def test_chunk_size_is_positive(self):
        """Vérifie que CHUNK_SIZE est positif."""
        assert CHUNK_SIZE > 0
        assert isinstance(CHUNK_SIZE, int)

    def test_chunk_overlap_is_positive(self):
        """Vérifie que CHUNK_OVERLAP est positif."""
        assert CHUNK_OVERLAP > 0
        assert isinstance(CHUNK_OVERLAP, int)

    def test_chunk_overlap_less_than_chunk_size(self):
        """Vérifie que le chevauchement est inférieur à la taille du chunk."""
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_ollama_base_url_format(self):
        """Vérifie le format de l'URL Ollama."""
        assert OLLAMA_BASE_URL.startswith("http://") or OLLAMA_BASE_URL.startswith("https://")
        assert "localhost" in OLLAMA_BASE_URL or "127.0.0.1" in OLLAMA_BASE_URL or "ollama" in OLLAMA_BASE_URL.lower()

    def test_ollama_model_is_string(self):
        """Vérifie que le modèle LLM est une chaîne non vide."""
        assert isinstance(OLLAMA_MODEL, str)
        assert len(OLLAMA_MODEL) > 0

    def test_embedding_model_is_string(self):
        """Vérifie que le modèle d'embedding est une chaîne non vide."""
        assert isinstance(EMBEDDING_MODEL, str)
        assert len(EMBEDDING_MODEL) > 0

    def test_chroma_path_is_path(self):
        """Vérifie que CHROMA_PATH est un objet Path."""
        assert isinstance(CHROMA_PATH, Path)

    def test_collection_name_is_string(self):
        """Vérifie que le nom de la collection est une chaîne non vide."""
        assert isinstance(COLLECTION_NAME, str)
        assert len(COLLECTION_NAME) > 0

    def test_pdf_path_is_path(self):
        """Vérifie que PDF_PATH est un objet Path."""
        assert isinstance(PDF_PATH, Path)

    def test_max_tokens_response_is_positive(self):
        """Vérifie que MAX_TOKENS_RESPONSE est positif."""
        assert MAX_TOKENS_RESPONSE > 0

    def test_temperature_is_in_valid_range(self):
        """Vérifie que la température est dans la plage valide."""
        assert 0.0 <= TEMPERATURE <= 1.0


class TestPaths:
    """Tests des chemins du projet."""

    def test_data_directory_exists(self):
        """Vérifie que le dossier data existe."""
        data_dir = BASE_DIR / "data"
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_src_directory_exists(self):
        """Vérifie que le dossier src existe."""
        src_dir = BASE_DIR / "src"
        assert src_dir.exists()
        assert src_dir.is_dir()

    def test_pdf_path_default_in_data(self):
        """Vérifie que le chemin PDF par défaut est dans le dossier data."""
        assert "data" in str(PDF_PATH).lower() or PDF_PATH.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])