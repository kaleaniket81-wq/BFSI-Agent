"""llm/src/ollama/provider.py — Ollama local LLM provider.

Models used:
  - Generation : llama3        (OLLAMA_LLM_MODEL)
  - Embeddings : nomic-embed-text (OLLAMA_EMBED_MODEL) — 768 dimensions
"""
import os
import requests
from typing import List
from llm.src.provider.base import LLMProvider


class OllamaProvider(LLMProvider):

    def __init__(self):
        self.base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm_model  = os.getenv("OLLAMA_LLM_MODEL", "llama3")
        self.embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def embed_dimensions(self) -> int:
        return 768   # nomic-embed-text output size

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """Call Ollama /api/chat with the llama3 model."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.llm_model,
                "messages": messages,
                "options": {"temperature": temperature},
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def embed(self, text: str) -> List[float]:
        """Generate 768-dim embedding via nomic-embed-text."""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def health(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
