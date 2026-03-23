"""
Module d'extraction du texte depuis le PDF.

Ce module gère la première étape du pipeline RAG:extraire le texte brut depuis le PDF.

Trois méthodes d'extraction sont disponibles (dans l'ordre de priorité):
1. Apache Tika: Serveur externe pour extraction PDF professionnelle
2. pdfplumber: Librairie Python pure (fallback si Tika non disponible)
3. Unstructured: Librairie tierce pour extraction avancée (optionnel)

Le résultat est une liste de dictionnaires contenant:
- page_number: Numéro de la page dans le PDF
- text: Texte brut extrait de la page
- metadata: Informations supplémentaires (source, page)

Ces données sont ensuite sauvegardées en JSON pour éviter de ré-extraire à chaque execution.
"""

from pathlib import Path
from typing import List, Dict
import json
import logging
import requests

from .config import PDF_PATH, TIKA_SERVER_URL

# Configuration du logging pour suivre l'exécution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf_tika(pdf_path: Path = None, tika_url: str = None) -> List[Dict[str, any]]:
    """
    Extrait le texte du PDF en utilisant Apache Tika.
    
    Apache Tika est un serveur qui analyse les fichiers et en extrait le contenu.
    Il est particulièrement efficace pour les PDF complexes avec plusieurs colonnes.
    
    Args:
        pdf_path: Chemin vers le fichier PDF (utilise PDF_PATH par défaut)
        tika_url: URL du serveur Tika (utilise TIKA_SERVER_URL par défaut)
    
    Returns:
        Liste de dictionnaires contenant le texte de chaque page
    """
    
    # Utiliser les valeurs par défaut si non fournies
    if pdf_path is None:
        pdf_path = PDF_PATH
    if tika_url is None:
        tika_url = TIKA_SERVER_URL
    
    # Vérifier que le fichier PDF existe
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF non trouvé: {pdf_path}")
    
    logger.info(f"Extraction du PDF via Tika: {pdf_path}")
    
    try:
        # Envoyer le PDF au serveur Tika
        with open(pdf_path, "rb") as f:
            headers = {
                "Content-Type": "application/pdf",
                "Accept": "application/json"
            }
            response = requests.put(
                f"{tika_url}/tika",
                data=f,
                headers=headers,
                timeout=120  # Timeout de 2 minutes
            )
        
        # Si Tika a réussi (code 200)
        if response.status_code == 200:
            content = response.json()
            # Extraire le texte (plusieurs clés possibles selon la configuration Tika)
            text = content.get("X-Tika-PDFOCR-Text", content.get("Content", ""))
            
            pages_data = []
            # Tika sépare parfois les pages avec ce motif
            page_texts = text.split("\n\n--- Page \n")
            
            if len(page_texts) > 1:
                # Plusieurs pages détectées
                for i, page_text in enumerate(page_texts[1:], start=1):
                    pages_data.append({
                        "page_number": i,
                        "text": page_text.strip(),
                        "metadata": {
                            "source": str(pdf_path.name),
                            "page": i
                        }
                    })
            else:
                # Une seule "page" ou contenu non paginé
                pages_data.append({
                    "page_number": 1,
                    "text": text.strip(),
                    "metadata": {
                        "source": str(pdf_path.name),
                        "page": 1
                    }
                })
            
            logger.info(f"Extraction terminée: {len(pages_data)} pages")
            return pages_data
        else:
            # Tika a retourné une erreur, utiliser le fallback
            logger.warning(f"Tika a retourné {response.status_code}, utilisation de pdfplumber comme fallback")
            return extract_text_from_pdf_fallback(pdf_path)
            
    except requests.exceptions.RequestException as e:
        # Tika non disponible (serveur pas démarré), utiliser le fallback
        logger.warning(f"Tika non disponible: {e}, utilisation de pdfplumber")
        return extract_text_from_pdf_fallback(pdf_path)


def extract_text_from_pdf_fallback(pdf_path: Path = None) -> List[Dict[str, any]]:
    """
    Fallback: Extrait le texte avec pdfplumber si Tika n'est pas disponible.
    
    pdfplumber est une librairie Python pure qui n'a pas besoin de serveur externe.
    Elle fonctionne bien pour la plupart des PDF mais peut avoir des difficultés
    avec les PDF complexes (multi-colonnes, images, etc.)
    """
    
    if pdf_path is None:
        pdf_path = PDF_PATH
    
    try:
        import pdfplumber
        
        logger.info(f"Extraction avec pdfplumber: {pdf_path}")
        
        pages_data = []
        
        # Ouvrir le PDF et lire chaque page
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                
                if text:  # Ne pas ajouter les pages vides
                    pages_data.append({
                        "page_number": page_num,
                        "text": text.strip(),
                        "metadata": {
                            "source": str(pdf_path.name),
                            "page": page_num
                        }
                    })
        
        logger.info(f"Extraction pdfplumber: {len(pages_data)} pages")
        return pages_data
        
    except ImportError:
        logger.error("pdfplumber non disponible non plus")
        raise ImportError("Veuillez installer pdfplumber ou démarrer Tika")


def extract_text_unstructured(pdf_path: Path = None) -> List[Dict[str, any]]:
    """
    Optionnel: Extrait le texte en utilisant Unstructured.
    
    Unstructured est une librairie tiers qui offre des fonctionnalités
    avancées d'extraction (gestion des tableaux, images, etc.)
    """
    
    if pdf_path is None:
        pdf_path = PDF_PATH
    
    try:
        from unstructured.partition.pdf import partition_pdf
        
        logger.info(f"Extraction avec Unstructured: {pdf_path}")
        
        # Extraire les éléments du PDF
        elements = partition_pdf(filename=str(pdf_path))
        
        pages_data = []
        current_page = 1
        current_text = ""
        
        for element in elements:
            # Récupérer le numéro de page si disponible
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
                page_num = element.metadata.page_number
            else:
                page_num = current_page
            
            # Changer de page
            if page_num != current_page and current_text:
                pages_data.append({
                    "page_number": current_page,
                    "text": current_text.strip(),
                    "metadata": {
                        "source": str(pdf_path.name),
                        "page": current_page
                    }
                })
                current_text = ""
                current_page = page_num
            
            current_text += "\n" + str(element)
        
        # Ajouter la dernière page
        if current_text:
            pages_data.append({
                "page_number": current_page,
                "text": current_text.strip(),
                "metadata": {
                    "source": str(pdf_path.name),
                    "page": current_page
                }
            })
        
        logger.info(f"Extraction Unstructured: {len(pages_data)} pages")
        return pages_data
        
    except ImportError as e:
        logger.warning(f"Unstructured non disponible: {e}")
        return extract_text_from_pdf_tika(pdf_path)


def extract_text_from_pdf(pdf_path: Path = None) -> List[Dict[str, any]]:
    """
    Point d'entrée principal pour l'extraction PDF.
    
    Essaie d'abord Tika, puis utilise pdfplumber comme fallback.
    Cette fonction est appelée par les autres modules du projet.
    
    Args:
        pdf_path: Chemin optionnel vers le PDF
    
    Returns:
        Liste de dictionnaires avec le texte de chaque page
    """
    return extract_text_from_pdf_tika(pdf_path)


def save_extracted_text(pages_data: List[Dict], output_path: Path = None):
    """
    Sauvegarde le texte extrait dans un fichier JSON.
    
    Cette étape est importante car elle permet de ne pas avoir à
    ré-extraire le PDF à chaque exécution. On charge simplement le JSON.
    
    Args:
        pages_data: Liste des pages extraites
        output_path: Chemin du fichier JSON de sortie
    """
    
    if output_path is None:
        output_path = PDF_PATH.parent / "data" / "processed" / "sanofi_text.json"
    
    # Créer le dossier parent si nécessaire
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder en JSON avec encodage UTF-8
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Texte extrait sauvegardé: {output_path}")


def load_extracted_text(input_path: Path = None) -> List[Dict]:
    """
    Charge le texte extrait depuis un fichier JSON.
    
    Args:
        input_path: Chemin du fichier JSON (utilise le fichier par défaut si None)
    
    Returns:
        Liste des pages extraites
    """
    
    if input_path is None:
        input_path = PDF_PATH.parent / "data" / "processed" / "sanofi_text.json"
    
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# POINT D'ENTRÉE (pour test direct)
# ============================================================================

if __name__ == "__main__":
    # Quand on execute ce fichier directement: extraire et sauvegarder
    pages = extract_text_from_pdf()
    save_extracted_text(pages)
    print(f"✓ {len(pages)} pages extraites")