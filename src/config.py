"""
Configuration du projet RAG Sanofi.

Ce fichier contient tous les paramètres de configuration du système.
Il utilise le fichier .env pour les variables d'environnement.

Explications des variables:
- OLLAMA_BASE_URL: URL du serveur Ollama (par défaut localhost:11434)
- OLLAMA_MODEL: Modèle LLM utilisé pour générer les réponses (qwen2.5:0.5b)
- EMBEDDING_MODEL: Modèle utilisé pour créer les vecteurs (nomic-embed-text:latest)
- CHROMA_PATH: Dossier où ChromaDB stocke les données
- COLLECTION_NAME: Nom de la collection dans ChromaDB
- PDF_PATH: Chemin vers le PDF source (rapport annuel Sanofi 2022)
- CHUNK_SIZE: Taille de chaque fragment de texte (en tokens)
- CHUNK_OVERLAP: Chevauchement entre les fragments (pour préserver le contexte)
- MAX_TOKENS_RESPONSE: Nombre maximum de tokens dans la réponse du LLM
- TEMPERATURE: Contrôle la créativité du LLM (0.0 = déterministe, 1.0 = créatif)
- TIKA_SERVER_URL: URL du serveur Apache Tika pour l'extraction PDF
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charge les variables d'environnement depuis le fichier .env
load_dotenv()

# Répertoire de base du projet
BASE_DIR = Path(__file__).parent.parent

# ============================================================================
# CONFIGURATION OLLAMA (Modèles LLM)
# ============================================================================

# URL du serveur Ollama (doit correspondre à "ollama serve" en cours d'exécution)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Modèle LLM utilisé pour générer les réponses
# Options: qwen2.5:0.5b, llama3.2, mistral:latest, gpt-oss:20b, gemma3:27b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

# Modèle utilisé pour créer les embeddings (vecteurs numériques)
# Ce modèle transforme le texte en vecteurs pour la recherche de similarité
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# ============================================================================
# CONFIGURATION CHROMADB (Base de données vectorielle)
# ============================================================================

# Dossier où ChromaDB stocke les données persistantes
CHROMA_PATH = BASE_DIR / os.getenv("CHROMA_PATH", "data/chroma")

# Nom de la collection dans ChromaDB
# Une collection = un ensemble de documents vectorisés
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sanofi_2022")

# ============================================================================
# CONFIGURATION PDF
# ============================================================================

# Chemin vers le fichier PDF source (rapport annuel Sanofi 2022)
# Par défaut, cherche dans le dossier data/ du projet
DEFAULT_PDF = BASE_DIR / "data" / "SANOFI-Integrated-Annual-Report-2022-EN.pdf"
PDF_PATH = Path(os.getenv("PDF_PATH", str(DEFAULT_PDF)))

# Dossier pour les données extraites (texte du PDF)
EXTRACTED_TEXT_PATH = BASE_DIR / "data" / "extracted_text.json"

# ============================================================================
# CONFIGURATION CHUNKING (Découpe du texte)
# ============================================================================

# Taille de chaque fragment de texte en tokens
# Plus le chunk est grand, plus il contient de contexte mais moins précis
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))

# Chevauchement entre les chunks pour préserver le contexte
# 200 tokens de chevauchement signifie que le chunk N et N+1 partagent 200 tokens
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ============================================================================
# CONFIGURATION LLM (Génération de réponse)
# ============================================================================

# Nombre maximum de tokens dans la réponse générée
MAX_TOKENS_RESPONSE = int(os.getenv("MAX_TOKENS_RESPONSE", "2000"))

# Température du LLM
# 0.0 = réponses déterministes et précises
# 0.3 = bon équilibre entre précision et créativité (recommandé)
# 1.0 = réponses très créatives mais moins fiables
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# ============================================================================
# CONFIGURATION TIKA (Extraction PDF)
# ============================================================================

# URL du serveur Apache Tika (optionnel, utilise pdfplumber si non disponible)
TIKA_SERVER_URL = os.getenv("TIKA_SERVER_URL", "http://localhost:9998")