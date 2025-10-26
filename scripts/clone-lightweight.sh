#!/bin/bash
# Lightweight clone script - excludes test files (80% smaller)
# Usage: curl -sSL https://raw.githubusercontent.com/bledden/arrwDB/main/scripts/clone-lightweight.sh | bash

set -e

echo "🚀 Cloning Vector Database API (lightweight - no tests)..."
git clone --filter=blob:none --sparse https://github.com/bledden/arrwDB.git
cd arrwDB

echo "📦 Configuring sparse checkout (excluding tests)..."
git sparse-checkout set '/*' '!tests'

echo "✅ Clone complete! (2,096 lines - 80% smaller than full repo)"
echo ""
echo "Next steps:"
echo "  cd arrwDB"
echo "  pip install -e ."
echo "  cp .env.example .env"
echo "  # Edit .env and add your COHERE_API_KEY"
echo "  python run_api.py"
