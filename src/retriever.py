"""Recherche sémantique dans les documents."""

from typing import List, Dict, Tuple
import logging

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Système de recherche sémantique."""
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """Récupère les documents les plus pertinents pour une requête."""
        
        logger.info(f"Recherche pour: '{query[:50]}...'")
        
        query_embedding = self.embedding_generator.encode_single(query)
        
        results = self.vector_store.search(query_embedding, n_results=top_k)
        
        documents = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            score = 1 - distance
            
            if score >= min_score:
                documents.append({
                    "text": results["documents"][0][i],
                    "page": results["metadatas"][0][i]["page"],
                    "source": results["metadatas"][0][i]["source"],
                    "score": score,
                    "chunk_id": results["metadatas"][0][i].get("chunk_id", i)
                })
        
        logger.info(f"{len(documents)} documents trouvés (score >= {min_score})")
        return documents
    
    def retrieve_with_expansion(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recherche avec expansion de requête (synonymes)."""
        
        query_expanded = query + " environnement carbone neutralité carbone objectifs"
        docs = self.retrieve(query_expanded, top_k=top_k)
        
        if not docs:
            docs = self.retrieve(query, top_k=top_k)
        
        return docs
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = None) -> List[Dict]:
        """Re-range les documents selon leur pertinence."""
        
        if top_k is None:
            top_k = len(documents)
        
        query_lower = query.lower()
        
        scored_docs = []
        for doc in documents:
            text_lower = doc["text"].lower()
            
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            
            overlap = len(query_words & text_words)
            relevance_score = doc["score"] + (overlap * 0.1)
            
            scored_docs.append((relevance_score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs[:top_k]]


def create_retriever(vector_store: VectorStore = None) -> Retriever:
    """Crée un retriever avec les composants nécessaires."""
    
    if vector_store is None:
        vector_store = VectorStore()
    
    embedding_generator = EmbeddingGenerator()
    
    return Retriever(vector_store, embedding_generator)


if __name__ == "__main__":
    from vector_store import load_vector_store
    
    store = load_vector_store()
    retriever = create_retriever(store)
    
    docs = retriever.retrieve("neutralité carbone environnement")
    for doc in docs:
        print(f"Page {doc['page']}: {doc['text'][:100]}... (score: {doc['score']:.2f})")