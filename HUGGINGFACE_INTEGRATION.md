# Hugging Face Integration Guide

This guide explains how to use the enhanced LLM API with support for both local models and Hugging Face models, including runtime switching between providers.

## Features

- **Dual Provider Support**: Use both local GGUF models and Hugging Face models
- **Runtime Switching**: Switch between providers without restarting the service
- **Unified API**: Same endpoints work with both providers
- **Provider Selection**: Choose provider per request or set a default
- **Model Management**: Load different Hugging Face models at runtime

## Configuration

### Environment Variables

Add these to your `.env` file or environment:

```bash
# Default provider (local or huggingface)
DEFAULT_PROVIDER=local

# Hugging Face settings
HUGGINGFACE_MODEL_NAME=microsoft/DialoGPT-medium
HUGGINGFACE_CACHE_DIR=./hf_cache

# Existing local model settings (unchanged)
MODEL_PATH=models/llama2-7b-q4.gguf
GPU_LAYERS=20
MODEL_CONTEXT_SIZE=2048
```

### Popular Hugging Face Models

Here are some recommended models for different use cases:

**Conversational Models:**
- `microsoft/DialoGPT-medium` - Good for conversations
- `facebook/blenderbot-400M-distill` - Facebook's conversational AI
- `microsoft/DialoGPT-small` - Lighter version for testing

**Instruction Following:**
- `google/flan-t5-small` - Google's instruction-tuned model
- `google/flan-t5-base` - Larger instruction model

**Code Generation:**
- `microsoft/CodeGPT-small-py` - Python code generation
- `Salesforce/codegen-350M-mono` - Multi-language code generation

## API Usage

### 1. Check Available Providers

```bash
curl http://localhost:8001/models/providers
```

Response:
```json
{
  "current_provider": "local",
  "available_providers": [
    {
      "name": "local",
      "display_name": "Local Model",
      "description": "Local GGUF model via llama-cpp-python",
      "model_path": "models/llama2-7b-q4.gguf",
      "context_size": 2048,
      "gpu_layers": 20
    },
    {
      "name": "huggingface",
      "display_name": "Hugging Face",
      "description": "Hugging Face model via transformers",
      "model_name": "microsoft/DialoGPT-medium",
      "device": "cuda",
      "torch_dtype": "torch.float16"
    }
  ]
}
```

### 2. Switch Provider at Runtime

```bash
# Switch to Hugging Face
curl -X POST http://localhost:8001/models/switch-provider \
  -H "Content-Type: application/json" \
  -d '{"provider": "huggingface"}'

# Switch to Hugging Face with specific model
curl -X POST http://localhost:8001/models/switch-provider \
  -H "Content-Type: application/json" \
  -d '{"provider": "huggingface", "huggingface_model_name": "google/flan-t5-small"}'

# Switch back to local
curl -X POST http://localhost:8001/models/switch-provider \
  -H "Content-Type: application/json" \
  -d '{"provider": "local"}'
```

### 3. Generate Text with Provider Selection

```bash
# Use current default provider
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing:",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Explicitly use Hugging Face
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing:",
    "provider": "huggingface",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Explicitly use local model
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing:",
    "provider": "local",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 4. Streaming with Provider Selection

```bash
# Streaming with Hugging Face
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "prompt": "Tell me a story about:",
    "provider": "huggingface",
    "stream": true,
    "max_tokens": 200
  }'
```

## Python Client Example

```python
import httpx
import asyncio

async def demo():
    async with httpx.AsyncClient() as client:
        # 1. Check available providers
        providers = await client.get("http://localhost:8001/models/providers")
        print("Available providers:", providers.json())
        
        # 2. Switch to Hugging Face
        switch = await client.post(
            "http://localhost:8001/models/switch-provider",
            json={"provider": "huggingface", "huggingface_model_name": "google/flan-t5-small"}
        )
        print("Switched to:", switch.json()["current_provider"])
        
        # 3. Generate with Hugging Face
        response = await client.post(
            "http://localhost:8001/generate",
            json={
                "prompt": "What is machine learning?",
                "provider": "huggingface",
                "max_tokens": 50,
                "temperature": 0.7
            }
        )
        result = response.json()
        print("Response:", result["text"])
        
        # 4. Switch back to local
        await client.post(
            "http://localhost:8001/models/switch-provider",
            json={"provider": "local"}
        )

asyncio.run(demo())
```

## Frontend Integration

The Vue.js frontend automatically works with both providers. You can add a provider selector:

```javascript
// Add to your frontend
const providers = await fetch('/models/providers').then(r => r.json());

// Provider selector in UI
<select v-model="selectedProvider">
  <option v-for="provider in providers.available_providers" 
          :key="provider.name" 
          :value="provider.name">
    {{ provider.display_name }}
  </option>
</select>

// Switch provider
async function switchProvider() {
  await fetch('/models/switch-provider', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider: selectedProvider })
  });
}

// Generate with specific provider
async function generateText() {
  const response = await fetch('/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: promptText,
      provider: selectedProvider, // Optional - uses current default if not specified
      max_tokens: 100
    })
  });
}
```

## Performance Considerations

### Local Models (llama-cpp-python)
- **Pros**: Fast inference, low latency, works offline
- **Cons**: Limited to downloaded models, requires GGUF format
- **Best for**: Production use, consistent performance

### Hugging Face Models (transformers)
- **Pros**: Access to thousands of models, easy experimentation
- **Cons**: Higher memory usage, slower initial load
- **Best for**: Development, model experimentation, research

### Memory Usage
- Local models: ~4-8GB VRAM for 7B models
- Hugging Face models: ~2-6GB VRAM for small models, 8-16GB for larger models
- Both models can be loaded simultaneously (uses more memory)

### Recommendations
1. Use local models for production workloads
2. Use Hugging Face for development and experimentation
3. Switch providers based on specific use cases
4. Monitor memory usage when both providers are loaded

## Troubleshooting

### Common Issues

1. **Hugging Face model fails to load**
   - Check internet connection
   - Verify model name exists on Hugging Face Hub
   - Ensure sufficient disk space for model cache
   - Check CUDA availability for GPU models

2. **Out of memory errors**
   - Reduce model size or use CPU-only models
   - Close other applications using GPU memory
   - Use model quantization (automatically handled)

3. **Slow inference**
   - Ensure GPU is being used (check device in model info)
   - Try smaller models for faster inference
   - Adjust batch size and context length

### Debug Endpoints

```bash
# Check health and provider status
curl http://localhost:8001/health

# Get detailed model information
curl http://localhost:8001/models/info

# Check available providers
curl http://localhost:8001/models/providers
```

## Example Use Cases

### 1. A/B Testing Models
```python
# Test same prompt with different providers
prompts = ["Explain AI:", "Write a poem:", "Solve this math problem:"]

for prompt in prompts:
    for provider in ["local", "huggingface"]:
        response = await generate_text(prompt, provider=provider)
        print(f"{provider}: {response['text']}")
```

### 2. Fallback Strategy
```python
async def generate_with_fallback(prompt):
    try:
        # Try Hugging Face first
        return await generate_text(prompt, provider="huggingface")
    except:
        # Fallback to local
        return await generate_text(prompt, provider="local")
```

### 3. Model Comparison
```python
# Compare responses from different models
models = [
    ("local", None),
    ("huggingface", "microsoft/DialoGPT-medium"),
    ("huggingface", "google/flan-t5-small")
]

for provider, model_name in models:
    if model_name:
        await switch_provider(provider, model_name)
    else:
        await switch_provider(provider)
    
    result = await generate_text("What is machine learning?")
    print(f"{provider}/{model_name}: {result['text']}")
```

This integration provides flexible, production-ready LLM inference with the ability to leverage both local optimization and the vast Hugging Face model ecosystem.
