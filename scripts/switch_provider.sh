#!/bin/bash
# scripts/switch_provider.sh — Hot-swap LLM provider without rebuilding
#
# Usage:
#   ./scripts/switch_provider.sh ollama     # local Llama 3 (default)
#   ./scripts/switch_provider.sh bedrock    # AWS Bedrock — Claude 3 Sonnet

PROVIDER=${1:-ollama}

if [[ "$PROVIDER" != "ollama" && "$PROVIDER" != "bedrock" ]]; then
  echo "Usage: $0 [ollama|bedrock]"
  echo ""
  echo "  ollama   — local Llama 3 via Ollama (free, private)"
  echo "  bedrock  — AWS Bedrock Claude 3 Sonnet (cloud, IAM auth)"
  exit 1
fi

# Update .env
sed -i "s/^LLM_PROVIDER=.*/LLM_PROVIDER=$PROVIDER/" .env
echo "LLM_PROVIDER switched to: $PROVIDER"

# Restart only the API + worker containers (no rebuild needed)
docker-compose restart api worker

echo ""
echo "Provider is now: $PROVIDER"

if [[ "$PROVIDER" == "bedrock" ]]; then
  echo ""
  echo "Requirements for AWS Bedrock:"
  echo "  1. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY set in .env"
  echo "     OR running on EC2/ECS with IAM role attached"
  echo "  2. Models enabled in AWS Console → Bedrock → Model access:"
  echo "     - anthropic.claude-3-sonnet-20240229-v1:0"
  echo "     - amazon.titan-embed-text-v2:0"
  echo "  3. AWS_REGION=ap-south-1 (Mumbai) set in .env"
fi
