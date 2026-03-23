# RAG Sanofi - Rapport Annuel 2022

## 🧠 Qu'est-ce que le RAG?

**RAG = Retrieval-Augmented Generation** (Génération Augmentée de Récupération)

C'est une technique qui permet à un modèle de langage (IA) de répondre à des questions en utilisant un document spécifique comme source de connaissances.

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Question  │ ───► │  RETRIEVAL   │ ───► │    LLM      │
│  utilisateur│      │ (Recherche)  │      │ (Génération)│
└─────────────┘      └──────────────┘      └─────────────┘
                           │                       │
                           ▼                       ▼
                    ChromaDB                 Réponse
                    (vecteurs)               + sources
```

**Comment ça marche?**
1. On transforme le PDF en vecteurs (embeddings) et on les stocke dans ChromaDB
2. Quand l'utilisateur pose une question, on recherche les passages les plus similaires
3. On envoie ces passages au LLM avec la question
4. Le LLM génère une réponse basée uniquement sur le contexte trouvé

---

## 📋 Projet - Questions du TP

Ce système répond automatiquement à ces 6 questions sur le rapport annuel Sanofi 2022:

| # | Question |
|---|----------|
| 1 | Objectifs environnementaux de Sanofi en matière de neutralité carbone et actions concrètes en 2022 |
| 2 | Avancées majeures du médicament Dupixent® et nouvelles indications approuvées |
| 3 | Mission de la fondation "Foundation S – The Sanofi Collective" et résultats en 2022 |
| 4 | Utilisation de l'intelligence artificielle dans la R&D (partenariats, projets) |
| 5 | Mesures pour promouvoir la diversité, l'équité et l'inclusion (DE&I) |
| 6 | Répartition des ventes de Sanofi 2022 par zone géographique et unité commerciale |

---

## 🛠️ Bibliothèques utilisées

| Bibliothèque | Role |
|--------------|------|
| **Langchain-Ollama** | Interface entre le code Python et Ollama (LLM local) |
| **ChromaDB** | Base de données vectorielle pour la recherche sémantique |
| **Streamlit** | Interface utilisateur web (le site que vous voyez) |
| **Unstructured** | Alternative pour l'extraction PDF |
| **Apache Tika** | Serveur d'extraction PDF professionnel |

---

## 📁 Structure du projet

```
rag_sanofi/
├── src/                          # Code source principal
│   ├── config.py                 # Configuration (variables d'environnement)
│   ├── extract_pdf.py            # Extraction du texte depuis le PDF
│   ├── chunking.py               # Découpe du texte en chunks
│   ├── vector_store.py           # Gestion de ChromaDB (stockage vectoriel)
│   ├── prompt_templates.py       # Prompts pour guider le LLM
│   └── rag_engine.py             # Moteur RAG (cerveau du système)
├── scripts/
│   ├── init_db.py                # Script pour initialiser la base de données
│   ├── process_pdf.py            # Script pour exécuter le pipeline complet
│   └── test_system.py            # Script pour tester le système
├── data/                         # Données
│   ├── raw/                      # PDF source
│   ├── processed/               # Texte extrait (JSON)
│   └── chroma/                  # Base de données vectorielle
├── streamlit_app.py             # Interface Streamlit
├── main.py                      # Interface CLI
├── api_main.py                  # Interface API REST
├── requirements.txt             # Liste des dépendances Python
└── .env                         # Variables de configuration
```

---

## 🚀 Installation et Lancement

### 1. Prérequis

- Python 3.12+
- Ollama installé et en cours d'exécution (`ollama serve`)
- Le PDF: `SANOFI-Integrated-Annual-Report-2022-EN.pdf`

### 2. Création du virtual environment et installation

```bash
cd /projet

# Créer le virtual environment
python3 -m venv .venv

# Activer le virtual environment
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Initialiser la base de données (une fois)

```bash
python scripts/init_db.py
```

Ce script:
- Extrait le texte du PDF (42 pages)
- Le découpe en chunks (108 fragments)
- Crée les vecteurs avec Ollama
- Stocke tout dans ChromaDB

### 4. Lancer le projet

#### Option A: Interface Streamlit (Recommandée)

```bash
streamlit run streamlit_app.py
```
Puis ouvrir: **http://localhost:8501**

#### Option B: Interface CLI (Ligne de commande)

```bash
python main.py --mode cli
```

Menu disponible:
1. Poser une question libre
2. Répondre aux 6 questions prédéfinies
3. Afficher les questions prédéfinies
4. Recherche sémantique (sans LLM)
5. Vérifier l'état du système
6. Quitter

#### Option C: API REST (Pour les développeurs)

```bash
python api_main.py
```

L'API est disponible sur: **http://localhost:8000**

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/` | GET | Route racine |
| `/health` | GET | État du système |
| `/predefined-questions` | GET | Liste des 6 questions |
| `/ask` | POST | Poser une question |
| `/answer-all` | POST | Répondre aux 6 questions |
| `/search` | GET | Recherche sémantique |

**Exemple:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les objectifs environnementaux?"}'
```

#### Option D: Export JSON

```bash
# Toutes les questions
python main.py --mode json --all-questions --output resultats.json

# Questions spécifiques
python main.py --mode json --question-ids 1 3 5
```

---

## ⚙️ Configuration

Le fichier `.env` contient tous les paramètres:

```env
# URL du serveur Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Modèle LLM pour les réponses
# Options: qwen2.5:0.5b, llama3.2, mistral:latest
OLLAMA_MODEL=qwen2.5:0.5b

# Modèle pour les embeddings (vecteurs)
EMBEDDING_MODEL=nomic-embed-text:latest

# Dossier pour ChromaDB
CHROMA_PATH=./data/chroma
COLLECTION_NAME=sanofi_2022

# Chemin vers le PDF source
PDF_PATH=/home/agouraud/Documents/Cours/Data IA/tp7/SANOFI-Integrated-Annual-Report-2022-EN.pdf

# Taille des chunks (en tokens)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

## ❓ Dépannage

### "Ollama pas accessible"
```bash
# Vérifier qu'Ollama est démarré
ollama serve

# Tester la connexion
curl http://localhost:11434/api/tags
```

### "Module non trouvé"
```bash
# Vérifier que le venv est activé
source .venv/bin/activate

# Réinstaller les dépendances
pip install -r requirements.txt
```

### "ChromaDB erreur"
```bash
# Supprimer et recréer la base
rm -rf data/chroma data/processed
python scripts/init_db.py
```

### Tester le système
```bash
python scripts/test_system.py
```

---

## 📊 Architecture technique

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTERFACE UTILISATEUR                       │
│  (Streamlit / CLI / API REST)                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG ENGINE (rag_engine.py)                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. RETRIEVAL: Recherche dans ChromaDB                     │  │
│  │    - Question → Embedding (vecteur)                      │  │
│  │    - Similarité dans la base de vecteurs                │  │
│  │    - Retourne les top_k chunks les plus similaires       │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ 2. AUGMENTATION: Construction du prompt                 │  │
│  │    - Contexte + Question + Template de prompt           │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ 3. GENERATION: Appel au LLM                              │  │
│  │    - Envoi du prompt à Ollama                            │  │
│  │    - Réception de la réponse                             │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐
│  ChromaDB       │  │  Langchain   │  │  Prompt         │
│  (vecteurs)     │  │  + Ollama    │  │  Templates      │
└─────────────────┘  └──────────────┘  └─────────────────┘
```

---

## 📦 Dépendances Python

```
chromadb>=0.4.0
langchain>=0.1.0
langchain-chroma>=0.1.0
langchain-community>=0.0.0
langchain-ollama>=0.0.0
streamlit>=1.30.0
unstructured>=0.12.0
tika>=2.6.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
rich>=13.0.0
httpx>=0.26.0
pandas>=2.0.0
fastapi>=0.109.0
uvicorn>=0.27.0
```

---

## 🔗 Ressources pour comprendre le RAG

- [Qu'est-ce que le RAG?](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Langchain Ollama](https://python.langchain.com/docs/integrations/llms/ollama/)
- [Ollama](https://ollama.com/)

---

## 📝 License

MIT - Projet éducatif pour le TP7 - Data IA
