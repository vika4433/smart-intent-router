#!/usr/bin/env python3
"""
Smart Intent Router System Validation Script
============================================

This script validates the entire Smart Intent Router system by testing:
1. Configuration loading and validation
2. Individual component functionality
3. MCP server connectivity
4. End-to-end routing functionality
5. Database connectivity (MongoDB)
6. LM Studio integration

Run this script after setting up the system to ensure everything is working correctly.
"""

import asyncio
import sys
import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import httpx
import aiohttp
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "smart-intent-router-server" / "src"))

class SystemValidator:
    def __init__(self):
        self.config_path = project_root / "config" / "smart_intent_router_config.yaml"
        self.config = None
        self.test_results = {}
        
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result with details."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"   {message}")
        self.test_results[test_name] = {"passed": passed, "message": message}
        
    def load_configuration(self) -> bool:
        """Test configuration loading and validation."""
        try:
            if not self.config_path.exists():
                self.log_result("Configuration File Exists", False, f"Config file not found: {self.config_path}")
                return False
                
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Validate required sections
            required_sections = ['intents', 'models', 'system_templates']
            for section in required_sections:
                if section not in self.config:
                    self.log_result("Configuration Structure", False, f"Missing section: {section}")
                    return False
                    
            # Validate models structure
            if not isinstance(self.config['models'], list) or len(self.config['models']) == 0:
                self.log_result("Model Configuration", False, "No models configured")
                return False
                
            # Check each model has required fields
            for i, model in enumerate(self.config['models']):
                required_fields = ['name', 'model_name', 'endpoint', 'supported_intents', 'supported_languages']
                for field in required_fields:
                    if field not in model:
                        self.log_result("Model Configuration", False, f"Model {i} missing field: {field}")
                        return False
                        
            self.log_result("Configuration Loading", True, f"Loaded {len(self.config['models'])} models")
            return True
            
        except Exception as e:
            self.log_result("Configuration Loading", False, str(e))
            return False
            
    def test_intent_classifier(self) -> bool:
        """Test intent classification functionality."""
        try:
            from intent_classifier.intent_classifier import IntentClassifier
            
            classifier = IntentClassifier(self.config)
            
            # Test various inputs
            test_cases = [
                ("Write a Python function to sort a list", "code"),
                ("What is 2 + 2?", "math"),
                ("Translate hello to Spanish", "translation"),
                ("Write a story about a dragon", "creative writing"),
                ("What is the capital of France?", "general")
            ]
            
            for message, expected_intent in test_cases:
                result = classifier.classify_intent(message)
                if result != expected_intent:
                    self.log_result("Intent Classification", False, 
                                  f"Expected '{expected_intent}' for '{message}', got '{result}'")
                    return False
                    
            self.log_result("Intent Classification", True, f"Tested {len(test_cases)} cases")
            return True
            
        except Exception as e:
            self.log_result("Intent Classification", False, str(e))
            return False
            
    def test_language_detector(self) -> bool:
        """Test language detection functionality."""
        try:
            from language_detector.language_detector import LanguageDetector
            
            detector = LanguageDetector()
            
            # Test various languages
            test_cases = [
                ("Hello, how are you?", "en"),
                ("Hola, ¿cómo estás?", "es"),
                ("Bonjour, comment ça va?", "fr"),
                ("Hallo, wie geht es dir?", "de"),
                ("Привет, как дела?", "ru")
            ]
            
            for message, expected_lang in test_cases:
                result = detector.detect_language(message)
                if result != expected_lang:
                    self.log_result("Language Detection", False, 
                                  f"Expected '{expected_lang}' for '{message}', got '{result}'")
                    return False
                    
            self.log_result("Language Detection", True, f"Tested {len(test_cases)} languages")
            return True
            
        except Exception as e:
            self.log_result("Language Detection", False, str(e))
            return False
            
    def test_llm_selector(self) -> bool:
        """Test LLM selection logic including weight-based selection."""
        try:
            from llm_selector.llm_selector import LLMSelector
            
            selector = LLMSelector(self.config)
            
            # Test intent-based selection
            result = selector.select_model("code", "en")
            if not result:
                self.log_result("LLM Selection", False, "No model selected for code intent")
                return False
                
            # Test weight-based selection if multiple models support same intent
            code_models = [m for m in self.config['models'] if 'code' in m.get('supported_intents', [])]
            if len(code_models) > 1:
                # Should select model with highest weight
                expected_model = max(code_models, key=lambda x: x.get('weight', 0))
                if result['name'] != expected_model['name']:
                    self.log_result("LLM Selection", False, 
                                  f"Weight-based selection failed. Expected {expected_model['name']}, got {result['name']}")
                    return False
                    
            self.log_result("LLM Selection", True, f"Selected model: {result['name']}")
            return True
            
        except Exception as e:
            self.log_result("LLM Selection", False, str(e))
            return False
            
    async def test_lm_studio_connectivity(self) -> bool:
        """Test connectivity to LM Studio endpoints."""
        if not self.config:
            return False
            
        try:
            endpoints = set(model['endpoint'] for model in self.config['models'])
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                for endpoint in endpoints:
                    try:
                        # Test basic connectivity
                        health_url = endpoint.replace('/v1/chat/completions', '/health')
                        async with session.get(health_url) as response:
                            if response.status == 200:
                                self.log_result(f"LM Studio Connectivity ({endpoint})", True)
                            else:
                                self.log_result(f"LM Studio Connectivity ({endpoint})", False, 
                                              f"HTTP {response.status}")
                                return False
                    except Exception as e:
                        self.log_result(f"LM Studio Connectivity ({endpoint})", False, str(e))
                        return False
                        
            return True
            
        except Exception as e:
            self.log_result("LM Studio Connectivity", False, str(e))
            return False
            
    def test_mongodb_connectivity(self) -> bool:
        """Test MongoDB database connectivity."""
        try:
            # Default MongoDB connection
            client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
            
            # Test connection
            client.admin.command('ping')
            
            # Test database creation/access
            db = client.smart_intent_router_test
            collection = db.test_collection
            
            # Insert test document
            test_doc = {"test": "validation", "timestamp": "2024-01-01"}
            result = collection.insert_one(test_doc)
            
            # Clean up
            collection.delete_one({"_id": result.inserted_id})
            client.close()
            
            self.log_result("MongoDB Connectivity", True, "Connection and operations successful")
            return True
            
        except ServerSelectionTimeoutError:
            self.log_result("MongoDB Connectivity", False, "MongoDB server not accessible")
            return False
        except Exception as e:
            self.log_result("MongoDB Connectivity", False, str(e))
            return False
            
    async def test_mcp_server_connectivity(self) -> bool:
        """Test MCP server connectivity and basic functionality."""
        try:
            # Test if server is running
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:8050/")
                if response.status_code != 200:
                    self.log_result("MCP Server Connectivity", False, 
                                  f"Server not accessible: HTTP {response.status_code}")
                    return False
                    
            # Test MCP tools endpoint
            try:
                from mcp.client.sse import sse_client
                from mcp import ClientSession
                
                async with sse_client("http://localhost:8050/sse") as streams:
                    read_stream, write_stream = streams
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        
                        # List available tools
                        tools_result = await session.list_tools()
                        tool_names = [tool.name for tool in tools_result.tools]
                        
                        expected_tools = ['route_request', 'classify_intent', 'detect_language', 'select_llm_model']
                        for tool in expected_tools:
                            if tool not in tool_names:
                                self.log_result("MCP Server Tools", False, f"Missing tool: {tool}")
                                return False
                                
                        self.log_result("MCP Server Connectivity", True, f"Found {len(tool_names)} tools")
                        return True
                        
            except Exception as e:
                self.log_result("MCP Server Connectivity", False, f"MCP protocol error: {e}")
                return False
                
        except Exception as e:
            self.log_result("MCP Server Connectivity", False, str(e))
            return False
            
    async def test_end_to_end_routing(self) -> bool:
        """Test complete end-to-end message routing."""
        try:
            from mcp.client.sse import sse_client
            from mcp import ClientSession
            
            async with sse_client("http://localhost:8050/sse") as streams:
                read_stream, write_stream = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # Test routing request
                    test_message = "Write a Python function to calculate fibonacci numbers"
                    result = await session.call_tool(
                        "route_request",
                        arguments={"messages": [{"role": "user", "content": test_message}]}
                    )
                    
                    if not result.content:
                        self.log_result("End-to-End Routing", False, "No response received")
                        return False
                        
                    response_text = result.content[0].text
                    
                    # Basic validation - should contain code
                    if "def" not in response_text or "fibonacci" not in response_text.lower():
                        self.log_result("End-to-End Routing", False, "Response doesn't contain expected code")
                        return False
                        
                    self.log_result("End-to-End Routing", True, "Successfully routed and processed request")
                    return True
                    
        except Exception as e:
            self.log_result("End-to-End Routing", False, str(e))
            return False
            
    def generate_report(self) -> None:
        """Generate a comprehensive test report."""
        print("\n" + "="*60)
        print("SMART INTENT ROUTER VALIDATION REPORT")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDETAILED RESULTS:")
        print("-" * 40)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status}: {test_name}")
            if result['message']:
                print(f"   {result['message']}")
                
        print("\n" + "="*60)
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Your Smart Intent Router is ready to use.")
        else:
            print("⚠️  Some tests failed. Please check the configuration and dependencies.")
            
        # Recommendations
        print("\nRECOMMENDations:")
        if not self.test_results.get("MongoDB Connectivity", {}).get("passed", False):
            print("- Start MongoDB: brew services start mongodb-community")
        if not self.test_results.get("LM Studio Connectivity", {}).get("passed", False):
            print("- Start LM Studio and enable the API server")
        if not self.test_results.get("MCP Server Connectivity", {}).get("passed", False):
            print("- Start the MCP server: python smart-intent-router-server/src/mcp_server/server.py")
            
    async def run_all_tests(self) -> None:
        """Run all validation tests."""
        print("🧪 Starting Smart Intent Router System Validation...")
        print("="*60)
        
        # Configuration tests
        if not self.load_configuration():
            print("❌ Configuration loading failed. Cannot continue with other tests.")
            return
            
        # Component tests
        self.test_intent_classifier()
        self.test_language_detector()
        self.test_llm_selector()
        
        # External dependency tests
        self.test_mongodb_connectivity()
        await self.test_lm_studio_connectivity()
        
        # Integration tests
        await self.test_mcp_server_connectivity()
        await self.test_end_to_end_routing()
        
        # Generate final report
        self.generate_report()

async def main():
    """Main entry point for the validation script."""
    validator = SystemValidator()
    await validator.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
