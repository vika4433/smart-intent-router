#!/usr/bin/env python3
"""
Comprehensive test suite for Smart Intent Router
Tests both classify_intent_tool_enabled modes and all supported intents/languages
"""

import requests
import json
import time
import asyncio
from typing import Dict, List, Any
import yaml

# Configuration
SERVER_URL = "http://localhost:8000"
CONFIG_FILE = "/Users/brs026/Documents/work/ds/sources/smart-intent-router/config/smart_intent_router_config.yaml"

class TestCase:
    def __init__(self, name: str, user_message: str, expected_intent: str, expected_language: str, expected_model: str, description: str = ""):
        self.name = name
        self.user_message = user_message
        self.expected_intent = expected_intent
        self.expected_language = expected_language
        self.expected_model = expected_model
        self.description = description

class TestResults:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.results = []
    
    def add_result(self, test_name: str, passed: bool, details: Dict[str, Any]):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        
        self.results.append({
            "test_name": test_name,
            "passed": passed,
            "details": details
        })
    
    def print_summary(self):
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ✅")
        print(f"Failed: {self.failed_tests} ❌")
        print(f"Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        
        if self.failed_tests > 0:
            print(f"\n{'='*80}")
            print(f"FAILED TESTS:")
            print(f"{'='*80}")
            for result in self.results:
                if not result["passed"]:
                    print(f"❌ {result['test_name']}")
                    print(f"   Details: {result['details']}")
        
        print(f"\n{'='*80}")

# Test cases for different intents and languages
TEST_CASES = [
    # Code intent tests
    TestCase(
        name="code_python_english",
        user_message="Write a Python function to reverse a string",
        expected_intent="code",
        expected_language="en",
        expected_model="qwen2.5-coder-7b-instruct",
        description="Basic Python coding request in English"
    ),
    TestCase(
        name="code_algorithm_english", 
        user_message="How to implement bubble sort algorithm?",
        expected_intent="code",
        expected_language="en", 
        expected_model="qwen2.5-coder-7b-instruct",
        description="Algorithm explanation request"
    ),
    TestCase(
        name="code_hebrew",
        user_message="כתוב פונקציה בפייתון שמחזירה את המספר הגדול ביותר ברשימה",
        expected_intent="code", 
        expected_language="he",
        expected_model="qwen2.5-coder-7b-instruct",
        description="Python coding request in Hebrew"
    ),
    
    # Math intent tests
    TestCase(
        name="math_calculation_english",
        user_message="What is the square root of 144?",
        expected_intent="math",
        expected_language="en",
        expected_model="wizardmath-7b-v1.1",
        description="Basic math calculation"
    ),
    TestCase(
        name="math_problem_english",
        user_message="Solve the equation: 2x + 5 = 15",
        expected_intent="math",
        expected_language="en", 
        expected_model="wizardmath-7b-v1.1",
        description="Math equation solving"
    ),
    TestCase(
        name="math_hebrew",
        user_message="מה זה השורש הריבועי של 169?",
        expected_intent="math",
        expected_language="he",
        expected_model="wizardmath-7b-v1.1", 
        description="Math question in Hebrew"
    ),
    
    # Translation intent tests
    TestCase(
        name="translation_request_english",
        user_message="Translate 'Hello, how are you?' to Spanish",
        expected_intent="translation",
        expected_language="en",
        expected_model="llama-2-7b-chat-hf-function-calling-v2",
        description="Translation request from English"
    ),
    TestCase(
        name="translation_request_french",
        user_message="Traduire cette phrase en anglais: 'Bonjour, comment allez-vous?'",
        expected_intent="translation", 
        expected_language="fr",
        expected_model="llama-2-7b-chat-hf-function-calling-v2",
        description="Translation request in French"
    ),
    
    # Creative writing intent tests
    TestCase(
        name="creative_story_english",
        user_message="Write a short story about a robot learning to paint",
        expected_intent="creative writing",
        expected_language="en",
        expected_model="llama-2-7b-chat-hf-function-calling-v2",
        description="Creative story writing request"
    ),
    TestCase(
        name="creative_poem_english",
        user_message="Compose a poem about the ocean",
        expected_intent="creative writing",
        expected_language="en", 
        expected_model="llama-2-7b-chat-hf-function-calling-v2",
        description="Poetry writing request"
    ),
    TestCase(
        name="creative_email_english",
        user_message="Write a professional email to request a meeting",
        expected_intent="creative writing",
        expected_language="en",
        expected_model="llama-2-7b-chat-hf-function-calling-v2",
        description="Professional email writing"
    ),
    
    # General intent tests  
    TestCase(
        name="general_question_english",
        user_message="What is the capital of France?",
        expected_intent="general",
        expected_language="en",
        expected_model="llama-2-7b-chat-hf-function-calling-v2",
        description="General knowledge question"
    ),
    TestCase(
        name="general_conversation_english",
        user_message="Tell me about the history of computers",
        expected_intent="general",
        expected_language="en",
        expected_model="llama-2-7b-chat-hf-function-calling-v2", 
        description="General conversation topic"
    ),
    TestCase(
        name="general_hebrew",
        user_message="מה זה בינה מלאכותית?",
        expected_intent="general",
        expected_language="he",
        expected_model="llama-2-7b-chat-hf-function-calling-v2",
        description="General question in Hebrew"
    ),
]

def load_config():
    """Load the current configuration"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def update_config(classify_intent_tool_enabled: bool):
    """Update the configuration file"""
    config = load_config()
    config['orchestrator']['classify_intent_tool_enabled'] = classify_intent_tool_enabled
    
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📝 Updated config: classify_intent_tool_enabled = {classify_intent_tool_enabled}")

def restart_server():
    """Restart the server to pick up configuration changes"""
    import subprocess
    import os
    import signal
    import sys
    
    print("🔄 Restarting server...")
    
    # Kill existing server
    try:
        subprocess.run(["pkill", "-f", "main.py"], check=False)
        time.sleep(2)
    except:
        pass
    
    # Start new server in background using the current Python interpreter
    server_dir = "/Users/brs026/Documents/work/ds/sources/smart-intent-router/smart-intent-router-server"
    subprocess.Popen(
        [sys.executable, "src/main.py"],  # Use current Python interpreter
        cwd=server_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(30):
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server started successfully after {i+1} seconds")
                return True
        except:
            time.sleep(1)
    
    print("❌ Server failed to start")
    return False

def run_test_case(test_case: TestCase, mode: str) -> Dict[str, Any]:
    """Run a single test case and return results"""
    print(f"  🧪 Running: {test_case.name}")
    
    try:
        # Make request to the AI routing endpoint
        response = requests.get(
            f"{SERVER_URL}/route_request_ai_stream",
            params={
                "user_message": test_case.user_message,
                "user_id": "test_user",
                "max_iterations": 5
            },
            stream=True,
            timeout=120
        )
        
        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}: {response.text}",
                "expected": {
                    "intent": test_case.expected_intent,
                    "language": test_case.expected_language, 
                    "model": test_case.expected_model
                }
            }
        
        # Parse SSE stream to extract workflow information
        workflow_data = {
            "intent": None,
            "language": None,
            "model": None,
            "final_response": None,
            "error": None,
            "steps": []
        }
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    
                    if data.get("type") == "intent_classified":
                        workflow_data["intent"] = data.get("intent")
                        workflow_data["steps"].append("intent_classified")
                    
                    elif data.get("type") == "language_detected":
                        workflow_data["language"] = data.get("language") 
                        workflow_data["steps"].append("language_detected")
                    
                    elif data.get("type") == "function_call" and data.get("function_name") == "send_to_llm":
                        params = data.get("parameters", {})
                        workflow_data["model"] = params.get("model_name")
                        workflow_data["steps"].append("send_to_llm")
                    
                    elif data.get("type") == "final_response":
                        workflow_data["final_response"] = data.get("response", "")[:200] + "..."
                        workflow_data["steps"].append("final_response")
                    
                    elif data.get("type") == "final_result":
                        result_data = data.get("data", {})
                        if "error" in result_data:
                            workflow_data["error"] = result_data["error"]
                        elif "response" in result_data:
                            workflow_data["final_response"] = result_data["response"][:200] + "..."
                            workflow_data["steps"].append("final_result")
                        break
                    
                except json.JSONDecodeError:
                    continue
        
        # Validate results
        validation_results = {
            "intent_correct": workflow_data["intent"] == test_case.expected_intent,
            "language_correct": workflow_data["language"] == test_case.expected_language,
            "model_correct": workflow_data["model"] == test_case.expected_model,
            "has_response": bool(workflow_data["final_response"]),
            "has_error": bool(workflow_data["error"])
        }
        
        passed = (
            validation_results["intent_correct"] and
            validation_results["language_correct"] and 
            validation_results["model_correct"] and
            validation_results["has_response"] and
            not validation_results["has_error"]
        )
        
        return {
            "passed": passed,
            "workflow_data": workflow_data,
            "validation": validation_results,
            "expected": {
                "intent": test_case.expected_intent,
                "language": test_case.expected_language,
                "model": test_case.expected_model
            },
            "steps_completed": workflow_data["steps"]
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "expected": {
                "intent": test_case.expected_intent,
                "language": test_case.expected_language,
                "model": test_case.expected_model
            }
        }

def run_test_suite(mode: str, classify_intent_tool_enabled: bool) -> TestResults:
    """Run the complete test suite for a given mode"""
    print(f"\n{'='*80}")
    print(f"TESTING MODE: {mode}")
    print(f"classify_intent_tool_enabled: {classify_intent_tool_enabled}")
    print(f"{'='*80}")
    
    results = TestResults()
    
    for test_case in TEST_CASES:
        print(f"\n📋 Test: {test_case.name}")
        print(f"   Message: {test_case.user_message}")
        print(f"   Expected: {test_case.expected_intent} | {test_case.expected_language} | {test_case.expected_model}")
        
        test_result = run_test_case(test_case, mode)
        
        passed = test_result.get("passed", False)
        if "error" in test_result:
            passed = False
            print(f"   ❌ ERROR: {test_result['error']}")
        else:
            workflow = test_result["workflow_data"]
            validation = test_result["validation"]
            
            print(f"   Actual:   {workflow['intent']} | {workflow['language']} | {workflow['model']}")
            print(f"   Steps:    {' → '.join(workflow['steps'])}")
            
            if passed:
                print(f"   ✅ PASSED")
            else:
                print(f"   ❌ FAILED")
                if not validation["intent_correct"]:
                    print(f"      ❌ Intent: expected {test_case.expected_intent}, got {workflow['intent']}")
                if not validation["language_correct"]:
                    print(f"      ❌ Language: expected {test_case.expected_language}, got {workflow['language']}")
                if not validation["model_correct"]:
                    print(f"      ❌ Model: expected {test_case.expected_model}, got {workflow['model']}")
                if validation["has_error"]:
                    print(f"      ❌ Error: {workflow['error']}")
                if not validation["has_response"]:
                    print(f"      ❌ No final response received")
        
        results.add_result(f"{mode}_{test_case.name}", passed, test_result)
        
        # Small delay between tests
        time.sleep(1)
    
    return results

def main():
    """Main test runner"""
    print("🚀 Starting Comprehensive Smart Intent Router Tests")
    print(f"Server: {SERVER_URL}")
    print(f"Config: {CONFIG_FILE}")
    
    all_results = TestResults()
    
    # Test both modes
    modes = [
        ("4-step mode (classify_intent_tool_enabled=true)", True),
        ("3-step mode (classify_intent_tool_enabled=false)", False)
    ]
    
    for mode_name, classify_enabled in modes:
        # Update configuration
        update_config(classify_enabled)
        
        # Restart server
        if not restart_server():
            print(f"❌ Failed to restart server for {mode_name}")
            continue
        
        # Run tests for this mode
        mode_results = run_test_suite(mode_name, classify_enabled)
        
        # Add to overall results
        for result in mode_results.results:
            all_results.add_result(result["test_name"], result["passed"], result["details"])
    
    # Print final summary
    all_results.print_summary()
    
    print(f"\n🏁 Testing completed!")
    return all_results.failed_tests == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
