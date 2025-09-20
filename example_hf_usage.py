#!/usr/bin/env python3
"""
Example script demonstrating how to use the updated LLM API with Hugging Face support.

This script shows how to:
1. Check available providers
2. Switch between local and Hugging Face models
3. Generate text with different providers
4. Handle provider-specific configurations
"""

import asyncio
import httpx
import json
from typing import Dict, Any


class LLMAPIClient:
    """Client for interacting with the LLM API."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status and provider information."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/models/info")
            response.raise_for_status()
            return response.json()
    
    async def get_available_providers(self) -> Dict[str, Any]:
        """Get list of available providers."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/models/providers")
            response.raise_for_status()
            return response.json()
    
    async def switch_provider(self, provider: str, hf_model_name: str = None) -> Dict[str, Any]:
        """Switch to a different model provider."""
        payload = {"provider": provider}
        if hf_model_name:
            payload["huggingface_model_name"] = hf_model_name
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/models/switch-provider",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    async def generate_text(
        self, 
        prompt: str, 
        provider: str = None,
        stream: bool = False,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using the specified provider."""
        payload = {
            "prompt": prompt,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if provider:
            payload["provider"] = provider
        
        async with httpx.AsyncClient() as client:
            if stream:
                # For streaming, we'll get the full response
                response = await client.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    headers={"Accept": "text/event-stream"}
                )
                response.raise_for_status()
                return {"streaming_response": response.text}
            else:
                response = await client.post(
                    f"{self.base_url}/generate",
                    json=payload
                )
                response.raise_for_status()
                return response.json()


async def main():
    """Main example function."""
    client = LLMAPIClient()
    
    print("🚀 LLM API with Hugging Face Support Demo")
    print("=" * 50)
    
    try:
        # 1. Check health and available providers
        print("\n1. Checking health and available providers...")
        health = await client.get_health()
        print(f"   Status: {health['status']}")
        print(f"   Current Provider: {health['current_provider']}")
        print(f"   Local Model Loaded: {health['local_model_loaded']}")
        print(f"   Hugging Face Model Loaded: {health['huggingface_model_loaded']}")
        
        providers = await client.get_available_providers()
        print(f"\n   Available Providers:")
        for provider in providers['available_providers']:
            print(f"   - {provider['display_name']}: {provider['description']}")
        
        # 2. Get detailed model information
        print("\n2. Getting detailed model information...")
        model_info = await client.get_model_info()
        print(f"   Current Provider: {model_info['current_provider']}")
        print(f"   Available Providers: {model_info['available_providers']}")
        
        if model_info['local_model_loaded']:
            print(f"   Local Model Path: {model_info['local_model_path']}")
            print(f"   Context Size: {model_info['context_size']}")
        
        if model_info['huggingface_model_loaded']:
            print(f"   Hugging Face Model: {model_info['huggingface_model_name']}")
            print(f"   Device: {model_info['device']}")
            print(f"   Torch Dtype: {model_info['torch_dtype']}")
        
        # 3. Test generation with current provider
        print("\n3. Testing text generation with current provider...")
        prompt = "The future of artificial intelligence is"
        
        result = await client.generate_text(
            prompt=prompt,
            stream=False,
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"   Prompt: {prompt}")
        print(f"   Response: {result['text']}")
        print(f"   Tokens Generated: {result['tokens_generated']}")
        print(f"   Generation Time: {result['generation_time']:.2f}s")
        
        # 4. Switch providers if both are available
        if len(providers['available_providers']) > 1:
            current_provider = health['current_provider']
            other_provider = "huggingface" if current_provider == "local" else "local"
            
            print(f"\n4. Switching from {current_provider} to {other_provider}...")
            switch_result = await client.switch_provider(other_provider)
            print(f"   Successfully switched to: {switch_result['current_provider']}")
            
            # Test generation with new provider
            print(f"\n5. Testing generation with {other_provider} provider...")
            result2 = await client.generate_text(
                prompt=prompt,
                stream=False,
                max_tokens=50,
                temperature=0.7
            )
            
            print(f"   Prompt: {prompt}")
            print(f"   Response: {result2['text']}")
            print(f"   Tokens Generated: {result2['tokens_generated']}")
            print(f"   Generation Time: {result2['generation_time']:.2f}s")
            
            # Switch back to original provider
            print(f"\n6. Switching back to {current_provider}...")
            await client.switch_provider(current_provider)
            print(f"   Successfully switched back to: {current_provider}")
        
        # 5. Test provider-specific generation
        print("\n7. Testing provider-specific generation...")
        
        # Generate with explicit provider selection
        for provider_name in ["local", "huggingface"]:
            if any(p['name'] == provider_name for p in providers['available_providers']):
                print(f"\n   Testing with {provider_name} provider:")
                try:
                    result = await client.generate_text(
                        prompt="Explain quantum computing in simple terms:",
                        provider=provider_name,
                        stream=False,
                        max_tokens=30,
                        temperature=0.5
                    )
                    print(f"   Response: {result['text']}")
                except Exception as e:
                    print(f"   Error with {provider_name}: {e}")
        
        print("\n✅ Demo completed successfully!")
        
    except httpx.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print("Make sure the API server is running on http://localhost:8001")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
