#!/bin/bash
# Run tests for all three LLM providers

# Try to load .env file if it exists
if [ -f .env ]; then
    echo "Loading API keys from .env file..."
    set -a
    source .env
    set +a
    echo ""
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "              TESTING ALL THREE LLM PROVIDERS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Check which API keys are set
echo "Checking API keys..."
echo ""

PROVIDERS=""

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✓ ANTHROPIC_API_KEY is set"
    PROVIDERS="${PROVIDERS} claude"
else
    echo "✗ ANTHROPIC_API_KEY not set (skipping Claude)"
fi

if [ -n "$GEMINI_API_KEY" ]; then
    echo "✓ GEMINI_API_KEY is set"
    PROVIDERS="${PROVIDERS} gemini"
else
    echo "✗ GEMINI_API_KEY not set (skipping Gemini)"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo "✓ OPENAI_API_KEY is set"
    PROVIDERS="${PROVIDERS} openai"
else
    echo "✗ OPENAI_API_KEY not set (skipping OpenAI)"
fi

if [ -z "$PROVIDERS" ]; then
    echo ""
    echo "Error: No API keys set!"
    echo ""
    echo "Set at least one:"
    echo "  export ANTHROPIC_API_KEY='your-key'"
    echo "  export GEMINI_API_KEY='your-key'"
    echo "  export OPENAI_API_KEY='your-key'"
    exit 1
fi

echo ""
echo "Running tests for:$PROVIDERS"
echo ""

# Run tests for each available provider
if [[ $PROVIDERS == *"claude"* ]]; then
    echo "───────────────────────────────────────────────────────────────────"
    echo "1. Testing Claude Haiku 4.5..."
    echo "───────────────────────────────────────────────────────────────────"
    python test_claude_generation.py
    echo ""
fi

if [[ $PROVIDERS == *"gemini"* ]]; then
    echo "───────────────────────────────────────────────────────────────────"
    echo "2. Testing Gemini 2.5 Flash..."
    echo "───────────────────────────────────────────────────────────────────"
    python test_gemini_generation.py
    echo ""
fi

if [[ $PROVIDERS == *"openai"* ]]; then
    echo "───────────────────────────────────────────────────────────────────"
    echo "3. Testing GPT-4o-mini..."
    echo "───────────────────────────────────────────────────────────────────"
    python test_openai_generation.py
    echo ""
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "              ALL TESTS COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Review the generated captions above"
echo "  2. If quality looks good, run full generation:"
for provider in $PROVIDERS; do
    echo "     python generate_data_with_${provider}.py"
done
echo "  3. Combine datasets:"
echo "     python combine_datasets.py"
echo ""

