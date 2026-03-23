"""
Module principal du moteur RAG.

Ce fichier orchestre tous les composants du système RAG:
1. Extraction PDF (extract_pdf.py)
2. Chunking (chunking.py)
3. Stockage vectoriel (vector_store.py)
4. Recherche sémantique (retriever.py)
5. Génération de réponse (Ollama via Langchain)

C'est le "cerveau" du système qui coordonne tout le flux de données.

Le flux de traitement d'une question:
1. Recevoir la question de l'utilisateur
2. Convertir la question en vecteur (embeddings)
3. Rechercher les chunks les plus similaires dans ChromaDB
4. Construire le prompt avec le contexte récupéré
5. Envoyer le prompt au LLM (Ollama)
6. Retourner la réponse + les sources utilisées
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE, MAX_TOKENS_RESPONSE
from .extract_pdf import extract_text_from_pdf, save_extracted_text, load_extracted_text
from .chunking import split_into_chunks
from .vector_store import VectorStore, init_vector_store, load_vector_store
from .prompt_templates import PREDEFINED_QUESTIONS, build_final_prompt

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Moteur RAG principal qui orchestre tous les composants.
    
    Cette classe gère le cycle de vie complet du système:
    - Initialisation (setup des composants)
    - Traitement des questions (ask)
    - Réponse aux questions prédéfinies (answer_all_predefined)
    - Recherche pure (search_only)
    - Vérification de l'état (health_check)
    
    Attributes:
        vector_store: Instance ChromaDB pour la recherche
        llm: Instance Ollama pour la génération de texte
    """
    
    def __init__(self):
        self.vector_store: Optional[VectorStore] = None
        self.llm: Optional[ChatOllama] = None
    
    def initialize(self, force_rebuild: bool = False):
        """
        Initialise tous les composants du RAG.
        
        Cette méthode doit être appelée avant toute utilisation du moteur.
        Elle:
        1. Crée le client Ollama (LLM)
        2. Charge ou crée la base de données vectorielle
        3. Vérifie que tout est opérationnel
        
        Args:
            force_rebuild: Si True, reconstruit l'index depuis le PDF
        """
        
        logger.info("Initialisation du moteur RAG avec Langchain...")
        
        # 1. Initialiser le LLM (Ollama via Langchain)
        self.llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS_RESPONSE
        )
        
        # 2. Charger la base de données vectorielle
        self.vector_store = load_vector_store()
        
        # 3. Si la base est vide, construire l'index
        if self.vector_store.count() == 0 or force_rebuild:
            logger.info("Base de données vide ou reconstruction demandée")
            self._build_index()
        
        # 4. Warmup du LLM - première requête pour charger le modèle en mémoire
        logger.info("Warmup du LLM...")
        self.llm.invoke("test")
        
        logger.info("Moteur RAG initialisé")
    
    def _build_index(self):
        """
        Construit l'index vectoriel (appelé automatiquement si nécessaire).
        
        Pipeline complet:
        1. Extraire le texte du PDF
        2. Découper en chunks
        3. Stocker dans ChromaDB
        """
        
        logger.info("Extraction du PDF...")
        pages = extract_text_from_pdf()
        save_extracted_text(pages)
        
        logger.info("Découpe en chunks...")
        chunks = split_into_chunks(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        logger.info("Sauvegarde dans ChromaDB...")
        self.vector_store = init_vector_store(chunks)
    
    def ask(self, question: str, category: str = None, top_k: int = 5) -> Dict:
        """
        Pose une question au système RAG.
        
        C'est la méthode principale utilisée pour répondre aux questions.
        Elle suit le流程:
        1. RETRIEVE: Rechercher les chunks pertinents
        2. GENERATE: Construire le prompt et appeler le LLM
        
        Args:
            question: Question de l'utilisateur
            category: Catégorie pour le prompt spécialisé (optionnel)
            top_k: Nombre de chunks à récupérer (5 par défaut)
        
        Returns:
            Dictionnaire avec:
            - question: La question posée
            - answer: La réponse générée
            - sources: Liste des sources utilisées
            - num_sources: Nombre de sources
        """
        
        # Vérifier que le moteur est initialisé
        if not self.vector_store or not self.llm:
            raise RuntimeError("Le moteur RAG n'est pas initialisé. Appelez initialize()")
        
        logger.info(f"Traitement de la question: {question[:50]}...")
        
        # ÉTAPE 1: RETRIEVAL (Recherche des chunks pertinents)
        # ---------------------------------------------------------
        # On cherche dans ChromaDB les chunks les plus similaires à la question
        # similarity_search_with_score retourne les +top_k avec leur score
        documents = self.vector_store.similarity_search_with_score(question, k=top_k)
        
        # Construire le contexte à partir des chunks trouvés
        context = "\n\n".join([
            f"[Page {doc['page']}] {doc['text']}"
            for doc in documents
        ])
        
        # ÉTAPE 2: PROMPT CONSTRUCTION (Construction du prompt)
        # ---------------------------------------------------------
        # Combiner le contexte, la catégorie et la question
        full_prompt = build_final_prompt(question, context, category)
        
        # ÉTAPE 3: GENERATION (Appel au LLM)
        # ---------------------------------------------------------
        logger.info("Génération de la réponse avec Langchain-Ollama...")
        
        # Créer la chaîne Langchain: Prompt -> LLM -> Output Parser
        prompt = ChatPromptTemplate.from_template(full_prompt)
        chain = prompt | self.llm | StrOutputParser()
        
        # Exécuter la chaîne
        answer = chain.invoke({})
        
        # Préparer les sources pour le retour
        sources = [
            {
                "page": doc["page"],
                "text": doc["text"][:200] + "...",  # Troncature pour l'affichage
                "score": doc["score"]
            }
            for doc in documents
        ]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(documents)
        }
    
    def answer_all_predefined(self, top_k: int = 5) -> List[Dict]:
        """
        Répond automatiquement aux 6 questions prédéfinies.
        
        Utile pour générer les réponses du rapport en une seule fois.
        
        Args:
            top_k: Nombre de chunks par question
        
        Returns:
            Liste des 6 réponses avec leurs sources
        """
        
        results = []
        
        for q in PREDEFINED_QUESTIONS:
            logger.info(f"Traitement de la question {q['id']}/{len(PREDEFINED_QUESTIONS)}")
            result = self.ask(q["question"], q["category"], top_k)
            result["question_id"] = q["id"]
            result["category"] = q["category"]
            results.append(result)
        
        return results
    
    def search_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Recherche sémantique uniquement (sans génération LLM).
        
        Utile pour explorer le contenu sans générer de réponse.
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats
        
        Returns:
            Liste des chunks pertinents
        """
        
        if not self.vector_store:
            raise RuntimeError("Le moteur RAG n'est pas initialisé")
        
        return self.vector_store.similarity_search_with_score(query, k=top_k)
    
    def health_check(self) -> Dict:
        """
        Vérifie l'état de tous les composants.
        
        Returns:
            Statut du système (ok ou degraded) avec détails
        """
        
        checks = {
            "ollama": False,
            "chroma": False
        }
        
        # Vérifier Ollama
        if self.llm:
            try:
                self.llm.invoke("test")  # Test simple
                checks["ollama"] = True
            except:
                pass
        
        # Vérifier ChromaDB
        if self.vector_store:
            checks["chroma"] = self.vector_store.count() > 0
        
        return {
            "status": "ok" if all(checks.values()) else "degraded",
            "checks": checks,
            "documents_count": self.vector_store.count() if self.vector_store else 0
        }


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def create_engine() -> RAGEngine:
    """Crée une nouvelle instance du moteur RAG."""
    return RAGEngine()


def initialize_and_answer(question: str, category: str = None) -> Dict:
    """
    Fonction utilitaire pour une réponse rapide (une seule question).
    
    Utile pour des tests rapides ou intégration simple.
    
    Args:
        question: Question à poser
        category: Catégorie optionnelle
    
    Returns:
        Réponse avec sources
    """
    engine = create_engine()
    engine.initialize()
    return engine.ask(question, category)


# ============================================================================
# POINT D'ENTRÉE (pour test direct)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test rapide du moteur
    engine = create_engine()
    engine.initialize()
    
    print("\n=== Test du moteur RAG ===")
    print(f"Documents dans la DB: {engine.vector_store.count()}")
    
    result = engine.ask("Quelle est la répartition des ventes de Sanofi en 2022?")
    print(f"\nRéponse: {result['answer'][:200]}...")