"""
Module des templates de prompts pour le LLM.

Ce module définit les prompts utilisés pour guider le modèle de langage.

Pourquoi utiliser des prompts spécialisés?
- Chaque type de question nécessite un contexte différent
- Le "system prompt" définit le persona du LLM (expert du rapport Sanofi)
- Le "user prompt"模板 combine le contexte récupéré et la question

Les 6 catégories correspondent aux 6 questions du TP:
1. neutralite_carbone: Objectifs environnementaux
2. dupixent: Développement du médicament
3. foundation_s: Fondation caritative
4. ia_rd: Intelligence artificielle en R&D
5. diversity: Diversité et inclusion
6. ventes: Répartition des ventes

Structure d'un prompt:
- System: définit le comportement du LLM
- User template: structure la réponse attendue
- Question: la vraie question de l'utilisateur
"""

from typing import Dict, List


# ============================================================================
# PROMPTS SYSTÈME (Persona du LLM)
# ============================================================================

SYSTEM_PROMPT = """Tu es un analyste expert du rapport annuel Sanofi 2022. 
Tu utilises uniquement les informations fournies dans le contexte pour répondre aux questions.
Sois précis, cite les sources (numéros de page) quand possible.
Si tu ne trouves pas l'information dans le contexte, dis-le clairement."""


# ============================================================================
# PROMPTS SPÉCIALISÉS PAR CATÉGORIE
# ============================================================================

PROMpts = {
    "neutralite_carbone": {
        "system": """Tu es un expert des objectifs environnementaux et de la durabilité chez Sanofi.
Réponds de manière détaillée en citant les actions concrètes entreprises en 2022.""",
        "user_template": """Contexte:
{context}

Question: Quels sont les objectifs environnementaux de Sanofi en matière de neutralité carbone et quelles actions concrètes ont été entreprises en 2022 pour y parvenir?

Réponds en français en utilisant uniquement les informations du contexte."""
    },
    
    "dupixent": {
        "system": """Tu es un expert du développement pharmaceutique et des approvals réglementaires.
Concentre-toi sur les avancées du médicament Dupixent®.""",
        "user_template": """Contexte:
{context}

Question: Quelles avancées majeures ont été réalisées dans le développement du médicament Dupixent® en 2022 et pour quelles nouvelles indications a-t-il été approuvé?

Réponds en français en utilisant uniquement les informations du contexte."""
    },
    
    "foundation_s": {
        "system": """Tu es un expert des activités philanthropiques et de responsabilité sociale de Sanofi.
Concentre-toi sur la fondation "Foundation S – The Sanofi Collective".""",
        "user_template": """Contexte:
{context}

Question: Quelle est la mission de la fondation "Foundation S – The Sanofi Collective", et quels résultats concrets a-t-elle obtenus en 2022?

Réponds en français en utilisant uniquement les informations du contexte."""
    },
    
    "ia_rd": {
        "system": """Tu es un expert de l'innovation et de la transformation digitale chez Sanofi.
Concentre-toi sur les applications de l'intelligence artificielle dans la R&D.""",
        "user_template": """Contexte:
{context}

Question: Comment Sanofi utilise-t-elle l'intelligence artificielle pour accélérer la recherche et le développement de nouveaux médicaments? Donne des exemples de partenariats ou de projets spécifiques.

Réponds en français en utilisant uniquement les informations du contexte."""
    },
    
    "diversity": {
        "system": """Tu es un expert des politiques RH et de gouvernance d'entreprise.
Concentre-toi sur la diversité, l'équité et l'inclusion (DE&I).""",
        "user_template": """Contexte:
{context}

Question: Quelles mesures Sanofi a-t-elle mises en place pour promouvoir la diversité, l'équité et l'inclusion (DE&I) dans ses effectifs et au sein de sa gouvernance en 2022?

Réponds en français en utilisant uniquement les informations du contexte."""
    },
    
    "ventes": {
        "system": """Tu es un expert des données financières et commerciales de Sanofi.
Sois précis sur les chiffres et les répartitions géographiques ou par unité commerciale.""",
        "user_template": """Contexte:
{context}

Question: Quelle est la répartition des ventes de Sanofi en 2022 par zone géographique et par unité commerciale?

Réponds en français en utilisant uniquement les informations du contexte. Donne des chiffres et pourcentages si disponibles."""
    }
}


# ============================================================================
# QUESTIONS PRÉDÉFINIES (Les 6 questions du TP)
# ============================================================================

PREDEFINED_QUESTIONS = [
    {
        "id": 1,
        "category": "neutralite_carbone",
        "question": "Quels sont les objectifs environnementaux de Sanofi en matière de neutralité carbone et quelles actions concrètes ont été entreprises en 2022 pour y parvenir ?"
    },
    {
        "id": 2,
        "category": "dupixent",
        "question": "Quelles avancées majeures ont été réalisées dans le développement du médicament Dupixent® en 2022 et pour quelles nouvelles indications a-t-il été approuvé ?"
    },
    {
        "id": 3,
        "category": "foundation_s",
        "question": "Quelle est la mission de la fondation \"Foundation S – The Sanofi Collective\", et quels résultats concrets a-t-elle obtenus en 2022 ?"
    },
    {
        "id": 4,
        "category": "ia_rd",
        "question": "Comment Sanofi utilise-t-elle l'intelligence artificielle pour accélérer la recherche et le développement de nouveaux médicaments ? Donne des exemples de partenariats ou de projets spécifiques."
    },
    {
        "id": 5,
        "category": "diversity",
        "question": "Quelles mesures Sanofi a-t-elle mises en place pour promouvoir la diversité, l'équité et l'inclusion (DE&I) dans ses effectifs et au sein de sa gouvernance en 2022 ?"
    },
    {
        "id": 6,
        "category": "ventes",
        "question": "Quelle est la répartition des ventes de Sanofi en 2022 par zone géographique et par unité commerciale ?"
    }
]


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_prompt_for_category(category: str, context: str) -> Dict[str, str]:
    """
    Récupère le prompt pour une catégorie donnée.
    
    Args:
        category: Nom de la catégorie (ex: "neutralite_carbone")
        context: Texte de contexte récupéré par le retriever
    
    Returns:
        Dictionnaire avec "system" et "user"
    """
    
    if category not in PROMpts:
        return {
            "system": SYSTEM_PROMPT,
            "user": f"Contexte:\n{context}\n\nQuestion: {{question}}\n\nRéponds en français."
        }
    
    template = PROMpts[category]["user_template"]
    return {
        "system": PROMpts[category]["system"],
        "user": template.format(context=context)
    }


def build_final_prompt(question: str, context: str, category: str = None) -> str:
    """
    Construit le prompt final à envoyer au LLM.
    
    Combine le system prompt, le user template et la question.
    
    Args:
        question: Question de l'utilisateur
        context: Texte récupéré par le retriever (passages pertinents)
        category: Catégorie optionnelle pour utiliser un prompt spécialisé
    
    Returns:
        Prompt complet prêt à être envoyé au LLM
    """
    
    if category and category in PROMpts:
        prompts = get_prompt_for_category(category, context)
        return f"{prompts['system']}\n\n{prompts['user']}\n\nQuestion: {question}"
    
    # Prompt générique si pas de catégorie
    return f"""{SYSTEM_PROMPT}

Contexte:
{context}

Question: {question}

Réponds en français en utilisant uniquement les informations du contexte."""


# ============================================================================
# POINT D'ENTRÉE (pour test direct)
# ============================================================================

if __name__ == "__main__":
    print("Questions prédéfinies:")
    for q in PREDEFINED_QUESTIONS:
        print(f"  {q['id']}. {q['question'][:60]}...")