"""
Script d'initialisation de la base de données vectorielle.

Ce script exécute le pipeline complet pour créer la base de données:
1. Extraction du texte depuis le PDF
2. Découpe en chunks
3. Génération des vecteurs (embeddings)
4. Stockage dans ChromaDB

Usage:
    python scripts/init_db.py           # Mode normal
    python scripts/init_db.py --force  # Forcer la reconstruction
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer les modules src
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.extract_pdf import extract_text_from_pdf, save_extracted_text
from src.chunking import split_into_chunks
from src.vector_store import init_vector_store
from src.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db(force: bool = False):
    """
    Initialise la base de données ChromaDB.
    
    Cette fonction guide l'utilisateur à travers le processus de création
    de l'index vectoriel à partir du PDF Sanofi.
    
    Args:
        force: Si True, reconstruction complète même si la DB existe
    """
    
    logger.info("=" * 60)
    logger.info("INITIALISATION DE LA BASE DE DONNÉES VECTORIELLE")
    logger.info("=" * 60)
    
    logger.info(f"PDF source: {PDF_PATH}")
    
    # Étape 1: Extraction du PDF
    logger.info("\n[1/4] 📄 Extraction du texte depuis le PDF...")
    logger.info("     Cette étape lit le fichier PDF et extrait le texte de chaque page.")
    pages = extract_text_from_pdf(PDF_PATH)
    print(f"  ✓ {len(pages)} pages extraites")
    
    # Sauvegarder le texte extrait (pour éviter de ré-extraire à chaque fois)
    save_extracted_text(pages)
    
    # Étape 2: Chunking
    logger.info("\n[2/4] ✂️ Découpe du texte en fragments (chunks)...")
    logger.info(f"     Taille: {CHUNK_SIZE} tokens, Chevauchement: {CHUNK_OVERLAP} tokens")
    chunks = split_into_chunks(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  ✓ {len(chunks)} fragments créés")
    
    # Étape 3: Stockage dans ChromaDB
    logger.info("\n[3/4] 💾 Stockage dans ChromaDB...")
    logger.info("     Conversion des fragments en vecteurs et索引ation...")
    store = init_vector_store(chunks)
    print(f"  ✓ {store.count()} documents indexés")
    
    # Résumé
    logger.info("\n" + "=" * 60)
    logger.info("✅ BASE DE DONNÉES INITIALISÉE AVEC SUCCÈS")
    logger.info("=" * 60)
    logger.info(f"   - Pages extraites: {len(pages)}")
    logger.info(f"   - Fragments créés: {len(chunks)}")
    logger.info(f"   - Vecteurs stockés: {store.count()}")
    logger.info(f"   - Emplacement: data/chroma/")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Initialise la base de données vectorielle RAG"
    )
    parser.add_argument(
        "--force", "-f", 
        action="store_true", 
        help="Force la reconstruction complète (supprime l'ancienne DB)"
    )
    
    args = parser.parse_args()
    
    # Si --force, supprimer l'ancienne base
    if args.force:
        import shutil
        chroma_path = Path(__file__).parent.parent / "data" / "chroma"
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            logger.info("Ancienne base supprimée")
    
    init_db(force=args.force)