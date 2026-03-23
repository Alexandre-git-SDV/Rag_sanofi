"""
Script pour exécuter le pipeline complet de traitement du PDF.

Ce script combine l'initialisation de la base de données et optionally
l'export des réponses aux questions.

Usage:
    python scripts/process_pdf.py                  # Juste créer la DB
    python scripts/process_pdf.py --export        # + export JSON
    python scripts/process_pdf.py --export --output resultats.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.extract_pdf import extract_text_from_pdf, save_extracted_text
from src.chunking import split_into_chunks
from src.vector_store import init_vector_store
from src.rag_engine import create_engine
from src.modes.json_export import export_to_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_full_pipeline(export_json: bool = False, output_path: str = None):
    """
    Exécute le pipeline complet de traitement.
    
    Étapes:
    1. Extraction du PDF
    2. Chunking
    3. Indexation ChromaDB
    4. Test du moteur RAG
    5. Export JSON (optionnel)
    
    Args:
        export_json: Si True, exporte les réponses en JSON
        output_path: Chemin du fichier de sortie JSON
    """
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLET DE TRAITEMENT")
    logger.info("=" * 60)
    
    # Étape 1: Extraction PDF
    logger.info("\n[1/5] 📄 Extraction du PDF...")
    from src.config import PDF_PATH
    pages = extract_text_from_pdf(PDF_PATH)
    save_extracted_text(pages)
    print(f"  ✓ {len(pages)} pages extraites")
    
    # Étape 2: Chunking
    logger.info("\n[2/5] ✂️ Découpe en chunks...")
    chunks = split_into_chunks(pages)
    print(f"  ✓ {len(chunks)} chunks créés")
    
    # Étape 3: Indexation ChromaDB
    logger.info("\n[3/5] 💾 Indexation dans ChromaDB...")
    store = init_vector_store(chunks)
    print(f"  ✓ {store.count()} documents indexés")
    
    # Étape 4: Test du moteur RAG
    logger.info("\n[4/5] 🧪 Test du moteur RAG...")
    engine = create_engine()
    engine.initialize()
    result = engine.ask("Quelle est la répartition des ventes de Sanofi?")
    print(f"  ✓ Réponse générée")
    print(f"  → {result['answer'][:150]}...")
    
    # Étape 5: Export JSON (optionnel)
    if export_json:
        logger.info("\n[5/5] 📦 Export JSON...")
        output = export_to_json(
            output_path=Path(output_path) if output_path else Path("results.json"),
            all_questions=True,
            top_k=5
        )
        print(f"  ✓ Export terminé: {output_path or 'results.json'}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Exécute le pipeline complet RAG")
    parser.add_argument(
        "--export", "-e", 
        action="store_true", 
        help="Exporter les réponses en JSON"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        help="Fichier de sortie JSON"
    )
    
    args = parser.parse_args()
    
    run_full_pipeline(
        export_json=args.export, 
        output_path=args.output
    )