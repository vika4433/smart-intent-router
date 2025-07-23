#!/usr/bin/env python3
"""
Test script to verify orchestrator model selection using is_orchestrator flag.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from mcp_server.server import get_orchestrator_model, get_models_from_config

async def test_orchestrator_selection():
    """Test the orchestrator model selection logic."""
    
    print("Testing orchestrator model selection...")
    print("=" * 50)
    
    # Test 1: Get all configured models
    print("1. Getting all configured models:")
    config_models = await get_models_from_config()
    for i, model in enumerate(config_models):
        is_orch = model.get("is_orchestrator", False)
        enabled = model.get("enabled", True)
        print(f"   {i+1}. {model.get('name')} ({model.get('model_name')}) - "
              f"Orchestrator: {is_orch}, Enabled: {enabled}")
    
    print()
    
    # Test 2: Get the orchestrator model
    print("2. Getting orchestrator model:")
    orchestrator = await get_orchestrator_model()
    if orchestrator:
        print(f"   Selected orchestrator: {orchestrator.get('name')} ({orchestrator.get('model_name')})")
        print(f"   Endpoint: {orchestrator.get('endpoint')}")
        print(f"   Source: {orchestrator.get('source')}")
    else:
        print("   ERROR: No orchestrator model found!")
    
    print()
    print("Test completed.")

if __name__ == "__main__":
    asyncio.run(test_orchestrator_selection())
