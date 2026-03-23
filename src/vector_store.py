"""
Module de gestion de la base de données vectorielle ChromaDB.

Ce module gère le stockage et la recherche des vecteurs (embeddings) dans ChromaDB.

Qu'est-ce que ChromaDB?
- Une base de données vectorielle open-source
- Optimisée pour la recherche de similarité (semantic search)
- Stocke des vecteurs associés à du texte et des métadonnées

Comment ça marche?
1. On ajoute des documents avec leurs embeddings
2. Lors d'une recherche, on converts la question en vecteur
3. ChromaDB trouve les vecteurs les plus similaires (plus proche voisins)

Integré avec Langchain:
- OllamaEmbeddings: pour générer les vecteurs via Ollama
- Chroma: l'interface de haut niveau pour ChromaDB
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from .config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Classe principale pour gérer la base de données vectorielle.
    
    Attributes:
        collection_name: Nom de la collection ChromaDB
        embeddings: Modèle d'embedding (génère les vecteurs)
        vectorstore: Instance ChromaDB pour les opérations
    """
    
    def __init__(self, collection_name: str = None):
        if collection_name is None:
            collection_name = COLLECTION_NAME
        
        # Créer le dossier si nécessaire
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        
        # Initialiser le modèle d'embedding (Ollama)
        # Ce modèle convertit le texte en vecteurs numériques
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        # Initialiser ChromaDB avec persistance
        # Les données sont sauvegardées sur le disque
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_PATH)
        )
        
        logger.info(f"Collection '{collection_name}' initialisée via Langchain")
    
    def add_documents(self, chunks: List[Dict]):
        """
        Ajoute les chunks à la collection ChromaDB.
        
        Chaque chunk est converti en vecteur et stocké avec ses métadonnées.
        
        Args:
            chunks: Liste des chunks à ajouter (issue de chunking.py)
        """
        
        # Convertir les chunks en Documents Langchain
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "page": chunk.get("page", 0),
                    "source": chunk.get("source", "unknown"),
                    "chunk_id": chunk.get("chunk_id", 0)
                }
            )
            documents.append(doc)
        
        # Ajouter les documents à ChromaDB
        # ChromaDB génère automatiquement les embeddings via OllamaEmbeddings
        self.vectorstore.add_documents(documents)
        
        logger.info(f"{len(chunks)} documents ajoutés à la collection")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Recherche les documents les plus similaires à une requête.
        
        Args:
            query: Question de l'utilisateur
            k: Nombre de résultats à retourner (5 par défaut)
        
        Returns:
            Liste des documents les plus pertinents
        """
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Dict]:
        """
        Recherche avec scores de similarité.
        
        Renvoie les documents avec leur score de similarité (0-1).
        Plus le score est proche de 0, plus le document est pertinent.
        
        Args:
            query: Question de l'utilisateur
            k: Nombre de résultats
        
        Returns:
            Liste de dictionnaires avec texte, page, source et score
        """
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        documents = []
        for doc, score in results:
            documents.append({
                "text": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
                "chunk_id": doc.metadata.get("chunk_id", 0)
            })
        
        return documents
    
    def delete_collection(self):
        """Supprime toute la collection (efface toutes les données)."""
        self.vectorstore.delete_collection()
        logger.info(f"Collection '{self.collection_name}' supprimée")
    
    def count(self) -> int:
        """Retourne le nombre de documents dans la collection."""
        return self.vectorstore._collection.count()


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def init_vector_store(chunks: List[Dict]) -> VectorStore:
    """
    Initialise une nouvelle base de données vectorielle.
    
    Args:
        chunks: Liste des chunks à indexer
    
    Returns:
        Instance VectorStore prête
    """
    store = VectorStore()
    store.add_documents(chunks)
    return store


def load_vector_store() -> VectorStore:
    """
    Charge une base de données vectorielle existante.
    
    Si la base n'existe pas, ChromaDB la crée automatiquement.
    
    Returns:
        Instance VectorStore
    """
    return VectorStore()


# ============================================================================
# POINT D'ENTRÉE (pour test direct)
# ============================================================================

if __name__ == "__main__":
    from extract_pdf import load_extracted_text
    from chunking import split_into_chunks
    
    # Exemple d'utilisation: charger les données et créer la DB
    pages = load_extracted_text()
    chunks = split_into_chunks(pages)
    
    store = init_vector_store(chunks)
    print(f"✓ Base de données initialisée avec {store.count()} documents")