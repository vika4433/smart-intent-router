#!/bin/bash

# Smart Intent Router Test Runner
# ==============================
# 
# This script runs the system validation tests to ensure all components
# are working correctly after setup.

set -e  # Exit on any error

echo "🧪 Smart Intent Router - System Test Runner"
echo "============================================"

# Check if we're in the right directory
if [ ! -f "config/smart_intent_router_config.yaml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected: config/smart_intent_router_config.yaml not found"
    exit 1
fi

# Check if Python virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating your venv: source venv/bin/activate"
fi

# Check dependencies
echo "📦 Checking Python dependencies..."
python -c "import yaml, pymongo, httpx, aiohttp, mcp" 2>/dev/null || {
    echo "❌ Missing dependencies. Please install requirements:"
    echo "   pip install -r requirements.txt"
    exit 1
}

echo "✅ Dependencies check passed"

# Run the validation script
echo ""
echo "🚀 Running system validation tests..."
python tests/test_system_validation.py

echo ""
echo "🏁 Test run complete!"
