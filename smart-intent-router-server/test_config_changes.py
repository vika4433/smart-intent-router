#!/usr/bin/env python3
"""
Simple test to verify orchestrator configuration changes.
"""

import yaml
import sys
from pathlib import Path

def test_config_changes():
    """Test that the configuration reflects the new orchestrator selection method."""
    
    config_path = Path(__file__).resolve().parent.parent / "config" / "smart_intent_router_config.yaml"
    
    print("Testing configuration changes...")
    print("=" * 50)
    
    # Load the configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test 1: Check models have is_orchestrator flag
    print("1. Checking models for is_orchestrator flag:")
    models = config.get("models", [])
    orchestrator_models = []
    
    for model in models:
        name = model.get("name")
        model_name = model.get("model_name")
        is_orch = model.get("is_orchestrator", False)
        enabled = model.get("enabled", True)
        
        print(f"   - {name} ({model_name}): is_orchestrator={is_orch}, enabled={enabled}")
        
        if is_orch and enabled:
            orchestrator_models.append(model)
    
    print()
    
    # Test 2: Check ai_router config
    print("2. Checking ai_router configuration:")
    ai_router = config.get("ai_router", {})
    
    # Check if old config keys are removed
    if "orchestrator_model" in ai_router:
        print("   WARNING: Old 'orchestrator_model' key still present in config!")
    else:
        print("   ✓ Old 'orchestrator_model' key removed")
    
    if "orchestrator_fallback_pattern" in ai_router:
        print("   WARNING: Old 'orchestrator_fallback_pattern' key still present in config!")
    else:
        print("   ✓ Old 'orchestrator_fallback_pattern' key removed")
    
    # Check system prompt exists
    if "system_prompt" in ai_router:
        print("   ✓ System prompt configured")
    else:
        print("   ERROR: System prompt missing!")
    
    print()
    
    # Test 3: Validate orchestrator selection logic
    print("3. Validating orchestrator selection logic:")
    
    if len(orchestrator_models) == 0:
        print("   WARNING: No models marked as orchestrator!")
        print("   The system should fallback to the first enabled model.")
        
        enabled_models = [m for m in models if m.get("enabled", True)]
        if enabled_models:
            print(f"   Fallback would be: {enabled_models[0].get('name')} ({enabled_models[0].get('model_name')})")
        else:
            print("   ERROR: No enabled models found!")
    
    elif len(orchestrator_models) == 1:
        orch = orchestrator_models[0]
        print(f"   ✓ One orchestrator model configured: {orch.get('name')} ({orch.get('model_name')})")
    
    else:
        print(f"   WARNING: Multiple orchestrator models found ({len(orchestrator_models)}). First one will be used:")
        for orch in orchestrator_models:
            print(f"     - {orch.get('name')} ({orch.get('model_name')})")
    
    print()
    print("Configuration test completed.")

if __name__ == "__main__":
    test_config_changes()
