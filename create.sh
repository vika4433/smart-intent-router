#!/bin/bash

# Create main directories
mkdir -p web-client/src/ui
mkdir -p web-client/src/mcp_client

mkdir -p smart-intent-router-server/src/mcp_server
mkdir -p smart-intent-router-server/src/language_detector
mkdir -p smart-intent-router-server/src/intent_classifier
mkdir -p smart-intent-router-server/src/llm_selector
mkdir -p smart-intent-router-server/src/lm_studio_proxy
mkdir -p smart-intent-router-server/src/response_handler
mkdir -p smart-intent-router-server/src/config
mkdir -p smart-intent-router-server/src/utils

mkdir -p tests
mkdir -p scripts
mkdir -p docs

# Create some key files
touch README.md LICENSE .gitignore

touch web-client/requirements.txt web-client/README.md
touch web-client/src/main.py

touch smart-intent-router-server/requirements.txt smart-intent-router-server/README.md
touch smart-intent-router-server/src/main.py
touch smart-intent-router-server/src/__init__.py

touch tests/test_intent_classifier.py
touch tests/test_language_detector.py
touch tests/test_llm_selector.py
touch tests/test_end_to_end.py

touch scripts/start_server.sh scripts/start_client.sh

touch docs/architecture.md docs/api_reference.md docs/user_guide.md

echo "Directory tree created!"
