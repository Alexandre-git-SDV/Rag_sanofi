"""Mode API FastAPI pour le RAG avec Langchain."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

from ..rag_engine import create_engine
from ..prompt_templates import PREDEFINED_QUESTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Sanofi API",
    description="API pour interroger le rapport annuel Sanofi 2022 via Langchain-Ollama",
    version="1.0.0"
)

engine = None


@app.on_event("startup")
async def startup_event():
    """Initialise le moteur RAG au démarrage."""
    global engine
    logger.info("Initialisation du moteur RAG...")
    engine = create_engine()
    engine.initialize()
    logger.info("Moteur RAG prêt")


class QuestionRequest(BaseModel):
    question: str
    category: Optional[str] = None
    top_k: Optional[int] = 5


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    num_sources: int
    question_id: Optional[int] = None
    category: Optional[str] = None


class BatchRequest(BaseModel):
    questions: List[str]
    top_k: Optional[int] = 5


@app.get("/")
async def root():
    """Route racine."""
    return {"message": "RAG Sanofi API - Rapport Annuel 2022"}


@app.get("/health")
async def health():
    """Vérifie l'état du système."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Moteur RAG non initialisé")
    return engine.health_check()


@app.get("/predefined-questions")
async def get_predefined_questions():
    """Retourne la liste des questions prédéfinies."""
    return {"questions": PREDEFINED_QUESTIONS}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Pose une question au système RAG."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Moteur RAG non initialisé")
    
    try:
        result = engine.ask(request.question, request.category, request.top_k)
        return QuestionResponse(**result)
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/batch")
async def ask_batch(request: BatchRequest):
    """Pose plusieurs questions en une fois."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Moteur RAG non initialisé")
    
    results = []
    for question in request.questions:
        try:
            result = engine.ask(question, top_k=request.top_k)
            results.append(result)
        except Exception as e:
            logger.error(f"Erreur pour la question '{question}': {e}")
            results.append({"error": str(e), "question": question})
    
    return {"results": results}


@app.post("/answer-all")
async def answer_all(top_k: int = 5):
    """Répond à toutes les questions prédéfinies."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Moteur RAG non initialisé")
    
    try:
        results = engine.answer_all_predefined(top_k)
        return {
            "total": len(results),
            "questions": [
                {
                    "question_id": r["question_id"],
                    "question": PREDEFINED_QUESTIONS[r["question_id"]-1]["question"],
                    "category": r["category"],
                    "answer": r["answer"],
                    "sources": r["sources"]
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search(query: str, top_k: int = 5):
    """Recherche sémantique directe (sans LLM)."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Moteur RAG non initialisé")
    
    try:
        results = engine.search_only(query, top_k)
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Crée l'application FastAPI."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)