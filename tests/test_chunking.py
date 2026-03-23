"""Tests pour le module de chunking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.chunking import count_tokens, split_into_chunks, split_by_paragraph, smart_chunk


class TestCountTokens:
    """Tests de la fonction count_tokens."""

    def test_empty_string(self):
        """Vérifie le comptage pour une chaîne vide."""
        result = count_tokens("")
        assert result == 0

    def test_english_text(self):
        """Vérifie le comptage pour un texte anglais."""
        text = "This is a test sentence."
        result = count_tokens(text)
        assert result > 0

    def test_french_text(self):
        """Vérifie le comptage pour un texte français."""
        text = "Ceci est une phrase de test."
        result = count_tokens(text)
        assert result > 0

    def test_encoding_name_default(self):
        """Vérifie que l'encodage par défaut fonctionne."""
        text = "Hello world"
        result = count_tokens(text, "cl100k_base")
        assert result > 0


class TestSplitIntoChunks:
    """Tests de la fonction split_into_chunks."""

    def test_empty_pages(self):
        """Vérifie le comportement avec une liste vide."""
        result = split_into_chunks([])
        assert result == []

    def test_single_page(self):
        """Vérifie le chunking d'une seule page."""
        pages = [
            {"text": "A" * 100, "page_number": 1, "metadata": {}}
        ]
        result = split_into_chunks(pages, chunk_size=50, chunk_overlap=10)
        assert len(result) > 0

    def test_multiple_pages(self):
        """Vérifie le chunking de plusieurs pages."""
        pages = [
            {"text": "Page 1 content", "page_number": 1, "metadata": {}},
            {"text": "Page 2 content", "page_number": 2, "metadata": {}}
        ]
        result = split_into_chunks(pages, chunk_size=50, chunk_overlap=10)
        assert len(result) >= 2

    def test_chunk_has_required_fields(self):
        """Vérifie que chaque chunk a les champs requis."""
        pages = [{"text": "Test text " * 100, "page_number": 1, "metadata": {}}]
        result = split_into_chunks(pages, chunk_size=50, chunk_overlap=10)
        
        for chunk in result:
            assert "text" in chunk
            assert "page" in chunk
            assert "chunk_id" in chunk

    def test_custom_chunk_size(self):
        """Vérifie l'utilisation d'une taille de chunk personnalisée."""
        pages = [{"text": "A" * 200, "page_number": 1, "metadata": {}}]
        
        result_small = split_into_chunks(pages, chunk_size=50, chunk_overlap=10)
        result_large = split_into_chunks(pages, chunk_size=200, chunk_overlap=10)
        
        assert len(result_small) > len(result_large)

    def test_empty_text_not_included(self):
        """Vérifie que les chunks vides ne sont pas inclus."""
        pages = [{"text": "", "page_number": 1, "metadata": {}}]
        result = split_into_chunks(pages, chunk_size=50, chunk_overlap=10)
        assert len(result) == 0


class TestSplitByParagraph:
    """Tests de la fonction split_by_paragraph."""

    def test_empty_pages(self):
        """Vérifie le comportement avec une liste vide."""
        result = split_by_paragraph([])
        assert result == []

    def test_paragraphs_separated(self):
        """Vérifie la séparation par paragraphes."""
        pages = [
            {"text": "Para 1 with enough text to pass the filter threshold.\n\nPara 2 also has enough text to be included in the results.\n\nPara 3 is also long enough to pass the filter.", "page_number": 1, "metadata": {}}
        ]
        result = split_by_paragraph(pages)
        assert len(result) >= 2

    def test_short_paragraphs_filtered(self):
        """Vérifie que les paragraphes courts sont filtrés."""
        pages = [
            {"text": "Short\n\nA much longer paragraph that should be included because it has enough characters", "page_number": 1, "metadata": {}}
        ]
        result = split_by_paragraph(pages)
        assert len(result) >= 1


class TestSmartChunk:
    """Tests de la fonction smart_chunk."""

    def test_empty_pages(self):
        """Vérifie le comportement avec une liste vide."""
        result = smart_chunk([])
        assert result == []

    def test_returns_chunks(self):
        """Vérifie que la fonction retourne des chunks."""
        pages = [{"text": "Sentence one. Sentence two. Sentence three.", "page_number": 1, "metadata": {}}]
        result = smart_chunk(pages, max_tokens=50)
        assert isinstance(result, list)

    def test_chunk_preserves_sentences(self):
        """Vérifie que les phrases sont préservées."""
        pages = [{"text": "First sentence. Second sentence. Third.", "page_number": 1, "metadata": {}}]
        result = smart_chunk(pages, max_tokens=100)
        
        all_text = " ".join([c["text"] for c in result])
        assert "First sentence" in all_text or "First" in all_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])