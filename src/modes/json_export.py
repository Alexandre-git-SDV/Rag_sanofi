"""Mode export JSON pour le RAG."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

from ..rag_engine import create_engine
from ..prompt_templates import PREDEFINED_QUESTIONS
from ..config import OLLAMA_MODEL

logger = logging.getLogger(__name__)


def export_to_json(
    output_path: Path = None,
    top_k: int = 5,
    model: str = None,
    all_questions: bool = False,
    question_ids: List[int] = None
) -> dict:
    """Exporte les réponses au format JSON."""
    
    if model is None:
        model = OLLAMA_MODEL
    
    if output_path is None:
        output_path = Path("results.json")
    
    logger.info("Initialisation du moteur RAG...")
    engine = create_engine()
    engine.initialize()
    
    logger.info("Génération des réponses...")
    
    metadata = {
        "report": "Sanofi Annual Report 2022",
        "date_generated": datetime.now().isoformat(),
        "model": model,
        "top_k": top_k,
        "using": "Langchain-Ollama + ChromaDB + Unstructured/Tika"
    }
    
    questions_to_answer = []
    
    if all_questions:
        questions_to_answer = PREDEFINED_QUESTIONS
    elif question_ids:
        questions_to_answer = [q for q in PREDEFINED_QUESTIONS if q["id"] in question_ids]
    else:
        questions_to_answer = PREDEFINED_QUESTIONS
    
    answers = []
    
    for q in questions_to_answer:
        logger.info(f"Traitement de la question {q['id']}...")
        result = engine.ask(q["question"], q["category"], top_k)
        
        answers.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": result["answer"],
            "sources": [
                {
                    "page": src["page"],
                    "text": src["text"],
                    "score": round(src["score"], 4)
                }
                for src in result["sources"]
            ],
            "num_sources": result["num_sources"]
        })
    
    output = {
        "metadata": metadata,
        "questions": answers,
        "summary": {
            "total_questions": len(answers),
            "model_used": model,
            "report_source": "SANOFI-Integrated-Annual-Report-2022-EN.pdf",
            "libraries": ["Langchain-Ollama", "ChromaDB", "Streamlit", "Unstructured", "Apache Tika"]
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Export terminé: {output_path}")
    
    return output


def print_summary(output: dict):
    """Affiche un résumé de l'export."""
    print("\n" + "="*60)
    print("RÉSUMÉ DE L'EXPORT JSON")
    print("="*60)
    print(f"Rapport: {output['metadata']['report']}")
    print(f"Date: {output['metadata']['date_generated']}")
    print(f"Modèle: {output['metadata']['model']}")
    print(f"Bibliothèques: {output['metadata']['using']}")
    print(f"Nombre de questions: {output['summary']['total_questions']}")
    print("-"*60)
    
    for q in output["questions"]:
        answer_preview = q["answer"][:100] + "..." if len(q["answer"]) > 100 else q["answer"]
        print(f"\n[{q['id']}] {q['category']}")
        print(f"Q: {q['question'][:60]}...")
        print(f"R: {answer_preview}")
        print(f"Sources: {q['num_sources']} documents")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export RAG vers JSON")
    parser.add_argument("--output", "-o", type=str, default="results.json", help="Fichier de sortie")
    parser.add_argument("--all", "-a", action="store_true", help="Exporter toutes les questions")
    parser.add_argument("--ids", "-i", type=int, nargs="+", help="IDs des questions à exporter")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Nombre de documents àretrieve")
    
    args = parser.parse_args()
    
    output = export_to_json(
        output_path=Path(args.output),
        all_questions=args.all or args.ids is None,
        question_ids=args.ids,
        top_k=args.top_k
    )
    
    print_summary(output)