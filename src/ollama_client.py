"""Client pour communiquer avec Ollama."""

import httpx
import json
import logging
from typing import Dict, Optional

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, MAX_TOKENS_RESPONSE, TEMPERATURE

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client pour communiquer avec Ollama en local."""
    
    def __init__(self, base_url: str = None, model: str = None):
        if base_url is None:
            base_url = OLLAMA_BASE_URL
        if model is None:
            model = OLLAMA_MODEL
        
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(timeout=120.0)
    
    def generate(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Génère une réponse à partir d'un prompt."""
        
        if temperature is None:
            temperature = TEMPERATURE
        if max_tokens is None:
            max_tokens = MAX_TOKENS_RESPONSE
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens
            }
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
        
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP: {e}")
            raise
    
    def generate_stream(self, prompt: str, temperature: float = None):
        """Génère une réponse en streaming."""
        
        if temperature is None:
            temperature = TEMPERATURE
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": True
        }
        
        try:
            with self.client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
        
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP: {e}")
            raise
    
    def chat(self, messages: list, temperature: float = None) -> str:
        """Génère une réponse en mode chat (plus récent)."""
        
        if temperature is None:
            temperature = TEMPERATURE
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
        
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP: {e}")
            raise
    
    def health(self) -> bool:
        """Vérifie si Ollama est disponible."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """Liste les modèles disponibles."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            data = response.json()
            return data.get("models", [])
        except:
            return []
    
    def close(self):
        """Ferme le client."""
        self.client.close()


def create_client() -> OllamaClient:
    """Crée un client Ollama."""
    return OllamaClient()


if __name__ == "__main__":
    client = OllamaClient()
    
    if client.health():
        print("✓ Ollama est disponible")
        models = client.list_models()
        print(f"Modèles disponibles: {[m['name'] for m in models]}")
    else:
        print("✗ Ollama n'est pas disponible")
    
    client.close()