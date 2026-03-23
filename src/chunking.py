"""
Module de chunking (découpe du texte en fragments).

Ce module transforme le texte brut (.pagesExtraites) en petits fragments appel chunks.
Chaque chunk est ensuite transformé en vecteur (embedding) pour la recherche de similarité.

Pourquoi decouper en chunks?
- Les modèles LLM ont une fenêtre de contexte limitée
- Les vecteurs sont plus petits et rapides à rechercher
- Permet de trouver les passages les plus pertinents pour chaque question

Deux stratégies principales:
- Decoupage par taille fixe avec chevauchement (utilisé par défaut)
- Decoupage intelligent par phrases (preserve le sens)
"""

import tiktoken
from typing import List, Dict
import logging

from .config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Compte le nombre de tokens dans un texte.
    
    Les tokens sont les unites de base que le LLM traite.
    En general, 1 token ~= 4 caractères en anglais, un peu plus en français.
    
    Args:
        text: Texte à analyser
        encoding_name: Type d'encodage (cl100k_base pour GPT-4/3.5)
    
    Returns:
        Nombre de tokens dans le texte
    """
    encoder = tiktoken.get_encoding(encoding_name)
    return len(encoder.encode(text))


def split_into_chunks(pages_data: List[Dict], chunk_size: int = None, chunk_overlap: int = None) -> List[Dict]:
    """
    Découpe le texte en chunks de taille fixe avec chevauchement.
    
    C'est la méthode par défaut. Elle coupe le texte tous les `chunk_size` tokens
    avec un chevauchement de `chunk_overlap` tokens pour préserver le contexte.
    
    Exemple avec chunk_size=100 et overlap=20:
    [0-100] [80-180] [160-260] ...
    
    Args:
        pages_data: Liste des pages avec leur texte (issue de extract_pdf.py)
        chunk_size: Taille de chaque chunk en tokens (1000 par défaut)
        chunk_overlap: Chevauchement entre chunks (200 par défaut)
    
    Returns:
        Liste de dictionnaires représentant les chunks
    """
    
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP
    
    chunks = []
    
    for page in pages_data:
        text = page["text"]
        page_num = page["page_number"]
        
        # Decouper page par page
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # Extraire le chunk
            chunk_text = text[start:end]
            
            # Ne pas ajouter les chunks vides
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "source": page.get("metadata", {}).get("source", "unknown"),
                    "chunk_id": len(chunks)
                })
            
            # Avancer avec le chevauchement
            start += chunk_size - chunk_overlap
    
    logger.info(f"Découpe terminée: {len(chunks)} chunks créés")
    return chunks


def split_by_paragraph(pages_data: List[Dict]) -> List[Dict]:
    """
    Découpe le texte par paragraphes (alternative au chunking par taille).
    
    Cette méthode préserve mieux la structure du texte mais peut créer
    des chunks de tailles très variables.
    """
    
    chunks = []
    
    for page in pages_data:
        text = page["text"]
        page_num = page["page_number"]
        
        # Séparer par double saut de ligne (paragraphes)
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            # Filtrer les paragraphes trop courts
            if para.strip() and len(para.strip()) > 50:
                chunks.append({
                    "text": para.strip(),
                    "page": page_num,
                    "source": page.get("metadata", {}).get("source", "unknown"),
                    "chunk_id": len(chunks)
                })
    
    logger.info(f"Découpe par paragraphes: {len(chunks)} chunks")
    return chunks


def smart_chunk(pages_data: List[Dict], max_tokens: int = None) -> List[Dict]:
    """
    Découpe intelligente basée sur les phrases.
    
    Cette méthode coupe au niveau des phrases (après les points)
    pour préserver le sens et éviter de couper un milieu de phrase.
    """
    
    if max_tokens is None:
        max_tokens = CHUNK_SIZE
    
    chunks = []
    
    for page in pages_data:
        text = page["text"]
        page_num = page["page_number"]
        
        # Séparer par phrases (après les points)
        sentences = text.split(". ")
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip() + ". "  # Rajouter le point
            
            # Si ça dépasse pas la limite, ajouter la phrase
            if count_tokens(current_chunk + sentence) <= max_tokens:
                current_chunk += sentence
            else:
                # Sauvegarder le chunk actuel et commencer un nouveau
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page_num,
                        "source": page.get("metadata", {}).get("source", "unknown"),
                        "chunk_id": len(chunks)
                    })
                current_chunk = sentence
        
        # Ajouter le dernier chunk de la page
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "page": page_num,
                "source": page.get("metadata", {}).get("source", "unknown"),
                "chunk_id": len(chunks)
            })
    
    logger.info(f"Découpe intelligente: {len(chunks)} chunks")
    return chunks


# ============================================================================
# POINT D'ENTRÉE (pour test direct)
# ============================================================================

if __name__ == "__main__":
    from extract_pdf import load_extracted_text
    
    pages = load_extracted_text()
    chunks = split_into_chunks(pages)
    print(f"✓ {len(chunks)} chunks créés")