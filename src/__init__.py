"""Module d'initialisation du package."""

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

__version__ = "1.0.0"
__all__ = [
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "EMBEDDING_MODEL",
    "CHROMA_PATH",
    "COLLECTION_NAME",
    "PDF_PATH",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
]