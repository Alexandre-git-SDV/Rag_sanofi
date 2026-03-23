"""Génération des embeddings pour le RAG."""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging
import os

from .config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


FALLBACK_MODELS = [
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "distiluse-base-multilingual-cased-v2"
]


def get_available_model(model_name: str = None) -> str:
    """Vérifie si le modèle est disponible, sinon utilise un fallback."""
    
    if model_name is None:
        model_name = EMBEDDING_MODEL
    
    for model in [model_name] + FALLBACK_MODELS:
        try:
            SentenceTransformer(model)
            logger.info(f"Modèle d'embedding utilisé: {model}")
            return model
        except Exception as e:
            logger.warning(f"Modèle {model} non disponible: {e}")
            continue
    
    raise RuntimeError("Aucun modèle d'embedding disponible")


class EmbeddingGenerator:
    """Générateur d'embeddings avec sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        available_model = get_available_model(model_name)
        
        logger.info(f"Chargement du modèle d'embedding: {available_model}")
        self.model = SentenceTransformer(available_model)
        self.model_name = available_model
        logger.info("Modèle d'embedding chargé")
    
    def encode(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """Encode une liste de textes en vecteurs d'embedding."""
        return self.model.encode(texts, show_progress_bar=show_progress)
    
    def encode_single(self, text: str) -> List[float]:
        """Encode un seul texte."""
        return self.model.encode(text, convert_to_numpy=True).tolist()


def generate_embeddings(chunks: List[Dict], model_name: str = None) -> List[Dict]:
    """Génère les embeddings pour tous les chunks."""
    
    generator = EmbeddingGenerator(model_name)
    
    texts = [chunk["text"] for chunk in chunks]
    
    logger.info(f"Génération des embeddings pour {len(texts)} chunks...")
    embeddings = generator.encode(texts, show_progress=True)
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    
    logger.info("Embeddings générés")
    return chunks


if __name__ == "__main__":
    from chunking import split_into_chunks
    from extract_pdf import load_extracted_text
    
    pages = load_extracted_text()
    chunks = split_into_chunks(pages)
    chunks_with_embeddings = generate_embeddings(chunks)
    print(f"✓ Embeddings générés pour {len(chunks_with_embeddings)} chunks")