#!/bin/bash
# Load environment variables from .env file

if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Copy .env.example to .env and add your API keys"
    exit 1
fi

# Load .env file
set -a
source .env
set +a

echo "✓ Environment variables loaded from .env"
echo ""
echo "API Keys status:"
[ -n "$ANTHROPIC_API_KEY" ] && echo "  ✓ ANTHROPIC_API_KEY is set" || echo "  ✗ ANTHROPIC_API_KEY not set"
[ -n "$GEMINI_API_KEY" ] && echo "  ✓ GEMINI_API_KEY is set" || echo "  ✗ GEMINI_API_KEY not set"
[ -n "$OPENAI_API_KEY" ] && echo "  ✓ OPENAI_API_KEY is set" || echo "  ✗ OPENAI_API_KEY not set"
echo ""

