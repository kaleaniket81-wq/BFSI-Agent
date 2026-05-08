"""
llm/src/bedrock/provider.py — AWS Bedrock cloud provider.

Replaces Azure OpenAI. Set LLM_PROVIDER=bedrock to activate.

Models used:
  Generation : amazon.titan-text-express-v1  OR  anthropic.claude-3-sonnet-20240229-v1:0
  Embeddings : amazon.titan-embed-text-v2:0  (1536-dim — same as ada-002)

AWS Bedrock advantages for BFSI:
  - Data stays in AWS region (ap-south-1 Mumbai — close to Pune)
  - IAM-based access — no API keys in code, uses AWS credentials
  - Pay-per-token — no monthly commitment
  - SOC2, ISO 27001, PCI DSS compliant — strong BFSI story

Interview line:
  "I replaced Azure OpenAI with AWS Bedrock for the cloud fallback.
   Both sit behind the same LLMProvider interface — zero code change in
   the RAG pipeline. AWS Bedrock uses IAM credentials so no API keys
   in environment variables, which is better security posture for BFSI."

Setup:
  1. Enable model access in AWS Console → Bedrock → Model access
     - Anthropic Claude 3 Sonnet
     - Amazon Titan Embeddings V2
  2. Set env vars (or use IAM role — recommended for EC2/ECS):
     AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION=ap-south-1
"""

import os
import json
from typing import List
from dotenv import load_dotenv

load_dotenv()


class BedrockProvider:
    """
    AWS Bedrock LLM provider.
    Uses boto3 — the same AWS SDK you already know from S3/RDS work.

    Generation model  : Anthropic Claude 3 Sonnet (best quality on Bedrock)
    Embedding model   : Amazon Titan Embeddings V2 (1536-dim, drop-in for ada-002)
    """

    # Model IDs — change here to switch models without touching anything else
    DEFAULT_LLM_MODEL   = "anthropic.claude-3-sonnet-20240229-v1:0"
    DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"

    def __init__(self):
        import boto3
        self.region = os.getenv("AWS_REGION", "ap-south-1")   # Mumbai

        # boto3 automatically reads AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
        # OR uses IAM role if running on EC2/ECS (recommended for production)
        self._client = boto3.client(
            service_name = "bedrock-runtime",
            region_name  = self.region,
        )

        self.llm_model   = os.getenv("BEDROCK_LLM_MODEL",   self.DEFAULT_LLM_MODEL)
        self.embed_model = os.getenv("BEDROCK_EMBED_MODEL",  self.DEFAULT_EMBED_MODEL)

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def embed_dimensions(self) -> int:
        return 1536   # Titan Embeddings V2 — same as ada-002, pgvector index unchanged

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """
        Call Claude 3 Sonnet on AWS Bedrock via InvokeModel API.
        Uses the Anthropic Messages API format (same as Claude.ai API).
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens":        2048,
            "temperature":       temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        if system:
            body["system"] = system

        response = self._client.invoke_model(
            modelId     = self.llm_model,
            body        = json.dumps(body),
            contentType = "application/json",
            accept      = "application/json",
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    def embed(self, text: str) -> List[float]:
        """
        Generate 1536-dim embedding via Amazon Titan Embeddings V2.
        Same dimensions as text-embedding-ada-002 — pgvector index works unchanged.
        """
        body = {
            "inputText":  text[:8192],   # Titan V2 max input
            "dimensions": 1536,
            "normalize":  True,          # unit vectors — better cosine similarity
        }

        response = self._client.invoke_model(
            modelId     = self.embed_model,
            body        = json.dumps(body),
            contentType = "application/json",
            accept      = "application/json",
        )

        result = json.loads(response["body"].read())
        return result["embedding"]

    def health(self) -> bool:
        """Check if Bedrock is reachable with current credentials."""
        try:
            import boto3
            client = boto3.client("bedrock", region_name=self.region)
            client.list_foundation_models(byProvider="Amazon")
            return True
        except Exception:
            return False
