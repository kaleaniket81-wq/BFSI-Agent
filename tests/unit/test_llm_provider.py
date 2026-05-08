"""tests/unit/test_llm_provider.py — Test LLM provider factory and Bedrock provider."""
import os
import pytest
from unittest.mock import patch, MagicMock


class TestLLMProviderFactory:

    def test_factory_returns_ollama_by_default(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            from llm.src.provider.base import LLMProviderFactory
            from llm.src.ollama.provider import OllamaProvider
            provider = LLMProviderFactory.get()
            assert isinstance(provider, OllamaProvider)

    def test_factory_returns_bedrock_when_configured(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER":            "bedrock",
            "AWS_ACCESS_KEY_ID":       "test-key",
            "AWS_SECRET_ACCESS_KEY":   "test-secret",
            "AWS_REGION":              "ap-south-1",
        }):
            with patch("boto3.client"):
                from llm.src.provider.base import LLMProviderFactory
                from llm.src.bedrock.provider import BedrockProvider
                provider = LLMProviderFactory.get()
                assert isinstance(provider, BedrockProvider)

    def test_factory_raises_on_unknown_provider(self):
        from llm.src.provider.base import LLMProviderFactory
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMProviderFactory.get("azure")   # azure removed — should raise

    def test_factory_raises_helpful_message(self):
        from llm.src.provider.base import LLMProviderFactory
        try:
            LLMProviderFactory.get("azure")
        except ValueError as e:
            assert "ollama" in str(e).lower()
            assert "bedrock" in str(e).lower()

    def test_ollama_provider_name(self):
        from llm.src.ollama.provider import OllamaProvider
        assert OllamaProvider().name == "ollama"

    def test_ollama_embed_dimensions(self):
        from llm.src.ollama.provider import OllamaProvider
        assert OllamaProvider().embed_dimensions == 768

    def test_bedrock_provider_name(self):
        with patch("boto3.client"):
            from llm.src.bedrock.provider import BedrockProvider
            assert BedrockProvider().name == "bedrock"

    def test_bedrock_embed_dimensions(self):
        with patch("boto3.client"):
            from llm.src.bedrock.provider import BedrockProvider
            assert BedrockProvider().embed_dimensions == 1536

    def test_ollama_complete_calls_api(self):
        from llm.src.ollama.provider import OllamaProvider
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Test answer"}}
        mock_response.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_response):
            result = OllamaProvider().complete("What is EMI?")
            assert result == "Test answer"

    def test_bedrock_complete_calls_invoke_model(self):
        with patch("boto3.client") as mock_boto:
            import json
            mock_client = MagicMock()
            mock_boto.return_value = mock_client
            mock_client.invoke_model.return_value = {
                "body": MagicMock(
                    read=lambda: json.dumps({
                        "content": [{"text": "Bedrock answer"}]
                    }).encode()
                )
            }
            from llm.src.bedrock.provider import BedrockProvider
            provider = BedrockProvider()
            result   = provider.complete("What is penalty?", system="You are a BFSI analyst.")
            assert result == "Bedrock answer"
            mock_client.invoke_model.assert_called_once()

    def test_bedrock_embed_calls_invoke_model(self):
        with patch("boto3.client") as mock_boto:
            import json
            mock_client = MagicMock()
            mock_boto.return_value = mock_client
            mock_client.invoke_model.return_value = {
                "body": MagicMock(
                    read=lambda: json.dumps({
                        "embedding": [0.1] * 1536
                    }).encode()
                )
            }
            from llm.src.bedrock.provider import BedrockProvider
            embedding = BedrockProvider().embed("Loan agreement text")
            assert len(embedding) == 1536
            assert embedding[0]   == 0.1

    def test_bedrock_embed_body_includes_normalization(self):
        """Titan V2 should use normalize=True for better cosine similarity."""
        with patch("boto3.client") as mock_boto:
            import json
            mock_client = MagicMock()
            mock_boto.return_value = mock_client
            mock_client.invoke_model.return_value = {
                "body": MagicMock(
                    read=lambda: json.dumps({"embedding": [0.0] * 1536}).encode()
                )
            }
            from llm.src.bedrock.provider import BedrockProvider
            BedrockProvider().embed("test text")
            call_body = json.loads(mock_client.invoke_model.call_args[1]["body"])
            assert call_body.get("normalize") is True
            assert call_body.get("dimensions") == 1536
