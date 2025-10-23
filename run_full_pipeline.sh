#!/bin/bash
# Complete automated pipeline: Generate data from 3 providers, combine, and train models

set -e  # Exit on error

# Load environment variables
if [ -f .env ]; then
    echo "Loading API keys from .env..."
    set -a
    source .env
    set +a
fi

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║           VORPAL COMPLETE TRAINING PIPELINE                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Check which providers are available
PROVIDERS=""
PROVIDER_COUNT=0

echo "Checking available API keys..."
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "  ✓ Claude Haiku 4.5 (Anthropic)"
    PROVIDERS="${PROVIDERS} claude"
    PROVIDER_COUNT=$((PROVIDER_COUNT + 1))
else
    echo "  ✗ Claude (no API key)"
fi

if [ -n "$GEMINI_API_KEY" ]; then
    echo "  ✓ Gemini 2.5 Flash (Google)"
    PROVIDERS="${PROVIDERS} gemini"
    PROVIDER_COUNT=$((PROVIDER_COUNT + 1))
else
    echo "  ✗ Gemini (no API key)"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo "  ✓ GPT-4o-mini (OpenAI)"
    PROVIDERS="${PROVIDERS} openai"
    PROVIDER_COUNT=$((PROVIDER_COUNT + 1))
else
    echo "  ✗ OpenAI (no API key)"
fi

if [ $PROVIDER_COUNT -eq 0 ]; then
    echo ""
    echo "❌ Error: No API keys found!"
    echo "Please set at least one API key in .env file"
    exit 1
fi

echo ""
echo "Will generate data from $PROVIDER_COUNT provider(s):$PROVIDERS"
echo ""

# Estimate costs and time
case $PROVIDER_COUNT in
    1)
        ESTIMATED_COST="$0.10-1.50"
        ESTIMATED_TIME="10-25 minutes"
        ESTIMATED_EXAMPLES="~8,440"
        ;;
    2)
        ESTIMATED_COST="$0.60-1.90"
        ESTIMATED_TIME="15-30 minutes"
        ESTIMATED_EXAMPLES="~16,880"
        ;;
    3)
        ESTIMATED_COST="$0.70-2.30"
        ESTIMATED_TIME="20-35 minutes"
        ESTIMATED_EXAMPLES="~25,320"
        ;;
esac

echo "Estimated cost: $ESTIMATED_COST"
echo "Estimated time: $ESTIMATED_TIME (parallel execution)"
echo "Expected output: $ESTIMATED_EXAMPLES examples"
echo ""

read -p "Proceed with full pipeline? (yes/no): " response
if [[ ! "$response" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "PHASE 1: DATA GENERATION (PARALLEL)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Create log directory
mkdir -p logs

# Start generation scripts in parallel
PIDS=()

if [[ $PROVIDERS == *"claude"* ]]; then
    echo "[1/3] Starting Claude data generation..."
    (echo "yes" | python generate_data_with_claude.py > logs/claude_generation.log 2>&1) &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
fi

if [[ $PROVIDERS == *"gemini"* ]]; then
    echo "[2/3] Starting Gemini data generation..."
    (echo "yes" | python generate_data_with_gemini.py > logs/gemini_generation.log 2>&1) &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
fi

if [[ $PROVIDERS == *"openai"* ]]; then
    echo "[3/3] Starting OpenAI data generation..."
    (echo "yes" | python generate_data_with_openai.py > logs/openai_generation.log 2>&1) &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
fi

echo ""
echo "All generators started in parallel (PIDs: ${PIDS[@]})"
echo "Monitoring progress..."
echo ""

# Monitor progress
START_TIME=$(date +%s)
while true; do
    RUNNING=0
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    
    ELAPSED=$(($(date +%s) - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    echo -ne "\r⏳ Generators running: $RUNNING/$PROVIDER_COUNT | Elapsed: ${MINUTES}m ${SECONDS}s"
    sleep 5
done

echo ""
echo ""

# Check if any failed
FAILED=0
for pid in "${PIDS[@]}"; do
    wait $pid
    if [ $? -ne 0 ]; then
        FAILED=$((FAILED + 1))
    fi
done

TOTAL_TIME=$(($(date +%s) - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "✓ All generators completed in ${MINUTES}m ${SECONDS}s"

if [ $FAILED -gt 0 ]; then
    echo "⚠️  Warning: $FAILED generator(s) had errors (check logs/)"
fi

echo ""

# Show generation logs summary
echo "Generation Summary:"
echo "───────────────────────────────────────────────────────────────────"
for provider in $PROVIDERS; do
    if [ -f "logs/${provider}_generation.log" ]; then
        echo ""
        echo "$provider:"
        tail -15 "logs/${provider}_generation.log" | grep -E "Saved|examples|complete|Error" | head -5
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "PHASE 2: COMBINING DATASETS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

python combine_datasets.py

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "PHASE 3: TRAINING MODELS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Update configs to use combined data
echo "Updating configs to use combined datasets..."

# Backup original configs
cp configs/config_coarse.yaml configs/config_coarse.yaml.backup 2>/dev/null || true
cp configs/config_fine.yaml configs/config_fine.yaml.backup 2>/dev/null || true

# Update train and valid paths in configs
python3 << 'PYEOF'
import yaml

# Update coarse config
with open('configs/config_coarse.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['train_path'] = 'data/train_combined.csv'
config['data']['valid_path'] = 'data/valid_combined.csv'

with open('configs/config_coarse.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

# Update fine config
with open('configs/config_fine.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['train_path'] = 'data/train_combined.csv'
config['data']['valid_path'] = 'data/valid_combined.csv'

with open('configs/config_fine.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("✓ Configs updated to use combined data")
PYEOF

echo ""
echo "[1/2] Training coarse model (categories)..."
echo "───────────────────────────────────────────────────────────────────"
python train.py --config configs/config_coarse.yaml

echo ""
echo "[2/2] Training fine model (subcategories)..."
echo "───────────────────────────────────────────────────────────────────"
python train.py --config configs/config_fine.yaml

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "PHASE 4: EVALUATION"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

echo "Evaluating coarse model..."
python eval.py --config configs/config_coarse.yaml --checkpoint models/model_coarse.vwbin

echo ""
echo "Evaluating fine model..."
python eval.py --config configs/config_fine.yaml --checkpoint models/model_fine.vwbin

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                  ✓✓✓ PIPELINE COMPLETE ✓✓✓                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

echo "Pipeline Statistics:"
echo "───────────────────────────────────────────────────────────────────"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Providers used: $PROVIDER_COUNT"
echo ""

echo "Generated Data:"
echo "  Training: data/train_combined.csv"
echo "  Validation: data/valid_combined.csv"
echo ""

echo "Trained Models:"
echo "  Coarse: models/model_coarse.vwbin"
echo "  Fine: models/model_fine.vwbin"
echo ""

echo "Logs:"
echo "  Generation logs: logs/*_generation.log"
echo "  API responses: data/generation_log*.jsonl"
echo ""

echo "Test inference:"
echo '  echo "football match highlights" | python inference.py \'
echo "    --config configs/config_coarse.yaml \\"
echo "    --checkpoint models/model_coarse.vwbin \\"
echo "    --stdin"
echo ""

