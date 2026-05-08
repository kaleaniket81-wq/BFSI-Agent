"""
llm/src/provider/base.py — Abstract LLM provider interface.

Switch providers with one env var:
    LLM_PROVIDER=ollama    (default — local, private, free)
    LLM_PROVIDER=bedrock   (AWS Bedrock — Claude 3 Sonnet + Titan Embeddings)

Interview line:
    "The LLM layer is fully provider-agnostic. Ollama runs locally for
     BFSI data privacy. AWS Bedrock is the cloud fallback — same interface,
     zero code change in the RAG pipeline, just one env-var switch."
"""
from abc import ABC, abstractmethod
from typing import List
import os


class LLMProvider(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """Generate a text completion."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/audit."""

    @property
    @abstractmethod
    def embed_dimensions(self) -> int:
        """Dimensionality of embedding vectors."""


class LLMProviderFactory:
    """Returns the correct provider based on LLM_PROVIDER env var."""

    @staticmethod
    def get(provider_name: str = None) -> LLMProvider:
        name = provider_name or os.getenv("LLM_PROVIDER", "ollama")

        if name == "ollama":
            from llm.src.ollama.provider import OllamaProvider
            return OllamaProvider()

        elif name == "bedrock":
            from llm.src.bedrock.provider import BedrockProvider
            return BedrockProvider()

        else:
            raise ValueError(
                f"Unknown LLM provider: '{name}'. "
                f"Valid options: 'ollama' (local) | 'bedrock' (AWS cloud)"
            )
