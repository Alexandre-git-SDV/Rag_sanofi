"""
Script de test pour vérifier que le système RAG fonctionne correctement.

Ce script teste les composants essentiels:
1. Import des modules
2. Configuration
3. Ollama (LLM)
4. ChromaDB (vecteurs)

Usage:
    python scripts/test_system.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test 1: Vérifier que tous les modules peuvent être importés."""
    print("\n" + "="*50)
    print("TEST 1: Import des modules")
    print("="*50)
    
    try:
        from src import config
        from src import extract_pdf
        from src import chunking
        from src import vector_store
        from src import prompt_templates
        from src import rag_engine
        print("✅ Tous les modules importés avec succès")
        return True
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False


def test_config():
    """Test 2: Vérifier la configuration."""
    print("\n" + "="*50)
    print("TEST 2: Configuration")
    print("="*50)
    
    try:
        from src.config import (
            OLLAMA_BASE_URL,
            OLLAMA_MODEL,
            EMBEDDING_MODEL,
            CHROMA_PATH,
            PDF_PATH
        )
        
        print(f"  Ollama URL: {OLLAMA_BASE_URL}")
        print(f"  Modèle LLM: {OLLAMA_MODEL}")
        print(f"  Modèle Embeddings: {EMBEDDING_MODEL}")
        print(f"  Chroma Path: {CHROMA_PATH}")
        print(f"  PDF Path: {PDF_PATH}")
        
        # Vérifier que le PDF existe
        if PDF_PATH.exists():
            print(f"  ✅ PDF trouvé: {PDF_PATH.name}")
        else:
            print(f"  ❌ PDF non trouvé: {PDF_PATH}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return False


def test_ollama():
    """Test 3: Vérifier qu'Ollama est accessible."""
    print("\n" + "="*50)
    print("TEST 3: Connexion Ollama")
    print("="*50)
    
    try:
        import httpx
        from src.config import OLLAMA_BASE_URL
        
        client = httpx.Client(timeout=10.0)
        response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"✅ Ollama accessible")
            print(f"   Modèles disponibles:")
            for model in models:
                print(f"   - {model['name']}")
            return True
        else:
            print(f"❌ Ollama a répondu: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Impossible de se connecter à Ollama:")
        print(f"   {e}")
        print(f"   Assurez-vous qu'Ollama est démarré: 'ollama serve'")
        return False


def test_chromadb():
    """Test 4: Vérifier ChromaDB."""
    print("\n" + "="*50)
    print("TEST 4: ChromaDB")
    print("="*50)
    
    try:
        from src.vector_store import load_vector_store
        
        store = load_vector_store()
        count = store.count()
        
        if count > 0:
            print(f"✅ ChromaDB opérationnelle")
            print(f"   Documents indexés: {count}")
        else:
            print(f"⚠️ ChromaDB vide - exécutez 'python scripts/init_db.py'")
        
        return True
    except Exception as e:
        print(f"❌ Erreur ChromaDB: {e}")
        return False


def test_rag_engine():
    """Test 5: Test du moteur RAG complet."""
    print("\n" + "="*50)
    print("TEST 5: Moteur RAG")
    print("="*50)
    
    try:
        from src.rag_engine import create_engine
        
        print("   Création du moteur...")
        engine = create_engine()
        
        print("   Initialisation...")
        engine.initialize()
        
        print("   Test de question...")
        result = engine.ask("Quelle est la répartition des ventes?")
        
        print(f"✅ Moteur RAG fonctionnel")
        print(f"   Réponse: {result['answer'][:100]}...")
        print(f"   Sources: {result['num_sources']} documents")
        
        return True
    except Exception as e:
        print(f"❌ Erreur moteur RAG: {e}")
        return False


def main():
    """Exécute tous les tests."""
    print("\n" + "="*60)
    print("🔍 TESTS DU SYSTÈME RAG SANOFI")
    print("="*60)
    
    results = []
    
    # Exécuter les tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Ollama", test_ollama()))
    results.append(("ChromaDB", test_chromadb()))
    results.append(("Moteur RAG", test_rag_engine()))
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 Tous les tests ont réussi!")
    else:
        print("⚠️ Certains tests ont échoué - vérifiez les erreurs ci-dessus")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()