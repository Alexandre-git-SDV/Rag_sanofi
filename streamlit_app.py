"""Interface Streamlit pour le RAG Sanofi."""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_engine import create_engine
from src.prompt_templates import PREDEFINED_QUESTIONS
from src.config import OLLAMA_MODEL

st.set_page_config(
    page_title="RAG Sanofi 2022",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 RAG Sanofi - Rapport Annuel 2022")
st.markdown("Interrogez le rapport annuel de Sanofi 2022 en langage naturel")


@st.cache_resource
def get_engine():
    """Initialise le moteur RAG une seule fois."""
    engine = create_engine()
    engine.initialize()
    return engine


if "engine" not in st.session_state:
    with st.spinner("Initialisation du moteur RAG (première chargement du modèle)..."):
        st.session_state.engine = get_engine()
    st.success("✓ Moteur RAG prêt")


st.sidebar.header("Configuration")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = OLLAMA_MODEL

model_name = st.sidebar.selectbox(
    "Modèle LLM",
    ["qwen2.5:0.5b", "llama3.2", "mistral", "gpt-oss:120b-cloud"],
    index=["qwen2.5:0.5b", "llama3.2", "mistral", "gpt-oss:120b-cloud"].index(st.session_state.selected_model) 
           if st.session_state.selected_model in ["qwen2.5:0.5b", "llama3.2", "mistral", "gpt-oss:120b-cloud"] else 0
)

top_k = st.sidebar.slider("Limite de documents/réponses", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.header("Questions rapides")

for q in PREDEFINED_QUESTIONS:
    if st.sidebar.button(f"Q{q['id']}: {q['category'].replace('_', ' ').title()}", key=f"q_{q['id']}"):
        st.session_state.current_question = q["question"]
        st.session_state.current_category = q["category"]


tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Questions prédéfinies", "🔍 Recherche"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Posez votre question sur le rapport annuel Sanofi 2022...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.spinner("Recherche et génération en cours..."):
            result = st.session_state.engine.ask(
                question,
                category=st.session_state.get("current_category"),
                top_k=top_k
            )
        
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            
            with st.expander("Sources utilisées"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**{i}.** Page {src['page']} (score: {src['score']:.2f})")
        
        st.session_state.current_question = None
        st.session_state.current_category = None

    if st.button("Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()


with tab2:
    st.header("6 Questions prédéfinies")
    
    for q in PREDEFINED_QUESTIONS:
        with st.expander(f"**{q['id']}.** {q['category'].replace('_', ' ').title()}"):
            st.markdown(f"_{q['question']}_")
            
            if st.button(f"Obtenir la réponse", key=f"answer_{q['id']}"):
                with st.spinner("Génération de la réponse..."):
                    result = st.session_state.engine.ask(q["question"], q["category"], top_k)
                    
                    st.markdown("### Réponse")
                    st.markdown(result["answer"])
                    
                    st.markdown("### Sources")
                    for i, src in enumerate(result["sources"], 1):
                        st.markdown(f"- Page {src['page']}: {src['text'][:100]}...")


with tab3:
    st.header("Recherche sémantique")
    
    query = st.text_input("Entrez une requête de recherche")
    
    if query and st.button("Rechercher"):
        results = st.session_state.engine.search_only(query, top_k=top_k)
        
        st.markdown(f"### {len(results)} résultats trouvés")
        
        for i, doc in enumerate(results, 1):
            with st.container():
                st.markdown(f"**Résultat {i}** - Page {doc['page']} (score: {doc['score']:.2f})")
                st.markdown(doc["text"][:500] + "...")
                st.divider()


st.markdown("---")
status = st.session_state.engine.health_check()
st.caption(f"Status: {status['status']} | Documents: {status['documents_count']} | Modèle: {st.session_state.selected_model}")