"""Tests pour le module d'embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

try:
    from src.embeddings import EmbeddingGenerator
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence_transformers non installé")
class TestEmbeddingGenerator:
    """Tests du générateur d'embeddings."""

    def test_embed_generator_instantiation(self):
        """Vérifie que le générateur peut être instancié."""
        generator = EmbeddingGenerator()
        assert generator is not None

    def test_encode_single_returns_list(self):
        """Vérifie que encode_single retourne une liste."""
        generator = EmbeddingGenerator()
        result = generator.encode_single("Test text")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_encode_single_returns_floats(self):
        """Vérifie que les embeddings sont des floats."""
        generator = EmbeddingGenerator()
        result = generator.encode_single("Test text")
        assert all(isinstance(x, float) for x in result)

    def test_encode_batch_returns_list_of_lists(self):
        """Vérifie que encode_batch retourne une liste de listes."""
        generator = EmbeddingGenerator()
        texts = ["Text 1", "Text 2", "Text 3"]
        result = generator.encode_batch(texts)
        assert isinstance(result, list)
        assert len(result) == len(texts)
        assert all(isinstance(x, list) for x in result)

    def test_same_text_produces_same_embedding(self):
        """Vérifie que le même texte produit le même embedding."""
        generator = EmbeddingGenerator()
        text = "Same text"
        result1 = generator.encode_single(text)
        result2 = generator.encode_single(text)
        assert result1 == result2

    def test_different_text_produces_different_embedding(self):
        """Vérifie que différents textes produisent différents embeddings."""
        generator = EmbeddingGenerator()
        result1 = generator.encode_single("Text A")
        result2 = generator.encode_single("Text B")
        assert result1 != result2

    def test_empty_text_handled(self):
        """Vérifie le comportement avec un texte vide."""
        generator = EmbeddingGenerator()
        result = generator.encode_single("")
        assert isinstance(result, list)


def test_embedding_generator_import():
    """Vérifie que le module peut être importé."""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        from src.embeddings import EmbeddingGenerator
        assert EmbeddingGenerator is not None
    else:
        pytest.skip("sentence_transformers non installé")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])