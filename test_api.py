#!/usr/bin/env python3
"""
Test script for the local LLM inference API.
Run this to verify your setup is working correctly.

This script provides comprehensive testing of all API endpoints and functionality.
"""

import asyncio
import json
import time
import os
import sys
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API configuration from environment
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_BASE = f"http://{API_HOST}:{API_PORT}"

# Test configuration
TEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

def test_health() -> Tuple[bool, Dict[str, Any]]:
    """Test health endpoint with retry logic."""
    logger.info("Testing health endpoint...")
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{API_BASE}/health", timeout=TEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health check passed: {data}")
                return True, data
            else:
                logger.warning(f"Health check failed (attempt {attempt + 1}): {response.status_code}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                return False, {}
        except Exception as e:
            logger.error(f"Health check error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            return False, {}
    
    return False, {}

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing model info...")
    try:
        response = requests.get(f"{API_BASE}/models/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: {data}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_non_streaming():
    """Test non-streaming generation"""
    print("\n🔍 Testing non-streaming generation...")
    
    payload = {
        "prompt": "Hello, how are you today?",
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/generate",
            json=payload,
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Non-streaming generation successful!")
            print(f"📝 Generated text: {data['text'][:100]}...")
            print(f"📊 Tokens: {data['tokens_generated']}")
            print(f"⏱️  Time: {data['generation_time']:.2f}s")
            print(f"🚀 Speed: {data['tokens_generated']/data['generation_time']:.1f} tokens/s")
            return True
        else:
            print(f"❌ Non-streaming generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Non-streaming generation error: {e}")
        return False

def test_streaming():
    """Test streaming generation"""
    print("\n🔍 Testing streaming generation...")
    
    payload = {
        "prompt": "Explain the concept of machine learning in simple terms.",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/generate",
            json=payload,
            timeout=60,
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Streaming generation started!")
            print("📝 Generated text:")
            
            full_text = ""
            token_count = 0
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])
                            if 'token' in data:
                                token = data['token']
                                full_text += token
                                token_count += 1
                                print(token, end='', flush=True)
                            elif 'event' in data and data['event'] == 'complete':
                                end_time = time.time()
                                print(f"\n\n✅ Streaming generation completed!")
                                print(f"📊 Total tokens: {token_count}")
                                print(f"⏱️  Total time: {end_time - start_time:.2f}s")
                                print(f"🚀 Average speed: {token_count/(end_time - start_time):.1f} tokens/s")
                                return True
                        except json.JSONDecodeError:
                            continue
            
            print(f"\n✅ Streaming completed! Generated {len(full_text)} characters")
            return True
        else:
            print(f"❌ Streaming generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Streaming generation error: {e}")
        return False

def test_caching():
    """Test Redis caching"""
    print("\n🔍 Testing caching...")
    
    payload = {
        "prompt": "What is the capital of France?",
        "max_tokens": 20,
        "stream": False
    }
    
    try:
        # First request (should cache)
        start_time = time.time()
        response1 = requests.post(f"{API_BASE}/generate", json=payload, timeout=30)
        time1 = time.time() - start_time
        
        # Second request (should be faster due to cache)
        start_time = time.time()
        response2 = requests.post(f"{API_BASE}/generate", json=payload, timeout=30)
        time2 = time.time() - start_time
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            if data1['text'] == data2['text']:
                print(f"✅ Caching test passed!")
                print(f"📊 First request: {time1:.2f}s")
                print(f"📊 Second request: {time2:.2f}s")
                print(f"🚀 Speed improvement: {time1/time2:.1f}x faster")
                return True
            else:
                print("❌ Cached response doesn't match original")
                return False
        else:
            print("❌ Caching test failed")
            return False
    except Exception as e:
        print(f"❌ Caching test error: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n🔍 Testing error handling...")
    
    # Test invalid request
    try:
        response = requests.post(
            f"{API_BASE}/generate",
            json={"invalid": "request"},
            timeout=10
        )
        if response.status_code == 422:  # Validation error
            print("✅ Error handling works correctly")
            return True
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error handling test error: {e}")
        return False

def run_performance_test():
    """Run a comprehensive performance test"""
    print("\n🚀 Running performance test...")
    
    test_prompts = [
        "Write a short story about a robot.",
        "Explain quantum computing.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "How does machine learning work?"
    ]
    
    total_tokens = 0
    total_time = 0
    successful_requests = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}/5: {prompt[:30]}...")
        
        payload = {
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{API_BASE}/generate", json=payload, timeout=60)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                total_tokens += data['tokens_generated']
                total_time += data['generation_time']
                successful_requests += 1
                
                print(f"✅ Generated {data['tokens_generated']} tokens in {data['generation_time']:.2f}s")
            else:
                print(f"❌ Request failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Request error: {e}")
    
    if successful_requests > 0:
        avg_tokens_per_second = total_tokens / total_time
        print(f"\n📊 Performance Summary:")
        print(f"   Successful requests: {successful_requests}/5")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average speed: {avg_tokens_per_second:.1f} tokens/s")
        
        # Performance expectations for RTX 4060
        if avg_tokens_per_second >= 15:
            print("🎉 Excellent performance!")
        elif avg_tokens_per_second >= 10:
            print("✅ Good performance!")
        elif avg_tokens_per_second >= 5:
            print("⚠️  Acceptable performance")
        else:
            print("❌ Performance below expectations")
    else:
        print("❌ No successful requests")

def main():
    """Run all tests"""
    print("🧪 Local LLM Inference API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Non-streaming Generation", test_non_streaming),
        ("Streaming Generation", test_streaming),
        ("Caching", test_caching),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is working correctly.")
        run_performance_test()
    else:
        print("❌ Some tests failed. Please check your setup.")
        print("\nTroubleshooting tips:")
        print("1. Make sure the API is running: python main.py")
        print("2. Check Redis is running: redis-server")
        print("3. Verify model file exists: ls -la models/llama2-7b-q4.gguf")
        print("4. Check GPU memory: nvidia-smi")

if __name__ == "__main__":
    main()
