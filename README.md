# Local LLM Inference Pipeline

A complete local inference solution for Llama-2-7b on RTX 4060 with 8GB VRAM. Features FastAPI streaming, Redis caching, and a beautiful Vue.js frontend.

## 🚀 Features

- **Local Inference**: Run Llama-2-7b locally with Q4_K_M quantization
- **Real-time Streaming**: Server-Sent Events for live token generation
- **Redis Caching**: Intelligent caching for improved performance
- **Web Interface**: Beautiful Vue.js frontend for testing
- **Docker Support**: Easy deployment with Docker Compose
- **Performance Monitoring**: Built-in metrics and health checks

## 📋 Requirements

- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or similar
- **CUDA**: 11.8+ with compatible drivers
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 10GB+ free space for models
- **Python**: 3.8+

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd llm-godo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download and Quantize Model

```bash
# Run the setup script (requires Hugging Face account)
python setup_model.py
```

**Manual Setup** (if script fails):
```bash
# 1. Login to Hugging Face
huggingface-cli login

# 2. Accept Llama 2 license
# Visit: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

# 3. Download model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-2-7b-chat-hf', cache_dir='./models/llama2-7b')
"

# 4. Convert to GGUF (requires llama.cpp)
# See: https://github.com/ggerganov/llama.cpp
```

### 3. Start Services

```bash
# Start Redis
redis-server --daemonize yes

# Start API
python main.py
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "stream": false}'
```

### 5. Use Web Interface

Open `frontend/index.html` in your browser or serve it:
```bash
cd frontend
python -m http.server 8080
# Visit: http://localhost:8080
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f llm-api
```

### Manual Docker Build

```bash
# Build image
docker build -t llm-inference .

# Run with GPU support
docker run --gpus all -p 8000:8000 -v $(pwd)/models:/app/models llm-inference
```

## 📊 Performance Expectations

### RTX 4060 (8GB VRAM) Benchmarks

| Metric | Value |
|--------|-------|
| **Tokens/second** | 15-20 tokens/s |
| **Latency** | 50-80ms per token |
| **Memory Usage** | ~6GB VRAM |
| **Context Window** | 2048 tokens |
| **Batch Size** | 512 tokens |

### Optimization Tips

1. **GPU Layers**: Adjust `GPU_LAYERS` environment variable (default: 20)
2. **Context Size**: Reduce `n_ctx` for lower memory usage
3. **Batch Size**: Increase `n_batch` for better throughput
4. **Quantization**: Use Q4_K_M for best speed/memory balance

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
MODEL_PATH=models/llama2-7b-q4.gguf
GPU_LAYERS=20

# Redis configuration
REDIS_URL=redis://localhost:6379/0

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### API Parameters

```json
{
  "prompt": "Your prompt here",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stream": true,
  "stop": ["</s>", "\n\n"]
}
```

## 📡 API Endpoints

### `POST /generate`
Generate text with optional streaming.

**Request:**
```json
{
  "prompt": "Explain quantum computing",
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": true
}
```

**Response (Streaming):**
```
data: {"token": "Quantum", "full_text": "Quantum", "token_count": 1}
data: {"token": " computing", "full_text": "Quantum computing", "token_count": 2}
...
data: {"event": "complete", "data": {"full_text": "...", "token_count": 50, "generation_time": 3.2}}
```

**Response (Non-streaming):**
```json
{
  "text": "Quantum computing is...",
  "tokens_generated": 50,
  "generation_time": 3.2
}
```

### `GET /health`
Check service health and model status.

### `GET /models/info`
Get model information and configuration.

## 🚀 Scaling Options

### For Higher Performance

1. **vLLM**: Use vLLM for better throughput
   ```bash
   pip install vllm
   # See: https://docs.vllm.ai/
   ```

2. **Cloud GPUs**: Deploy on AWS/GCP with larger GPUs
3. **Model Sharding**: Split model across multiple GPUs
4. **Batch Processing**: Process multiple requests simultaneously

### For Lower Resource Usage

1. **Smaller Models**: Use Llama-2-7b-instruct or smaller variants
2. **Lower Quantization**: Use Q2_K for 2-bit quantization
3. **CPU Fallback**: Run on CPU with more RAM
4. **Model Pruning**: Remove unnecessary model weights

## 🔍 Troubleshooting

### Common Issues

**Model Loading Errors:**
```bash
# Check CUDA installation
nvidia-smi

# Verify model path
ls -la models/llama2-7b-q4.gguf

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Memory Issues:**
```bash
# Reduce GPU layers
export GPU_LAYERS=15

# Reduce context size in main.py
n_ctx=1024  # instead of 2048
```

**Redis Connection Errors:**
```bash
# Start Redis manually
redis-server

# Check Redis status
redis-cli ping
```

### Performance Debugging

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check API logs
tail -f logs/api.log

# Test with curl
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "stream": false}' \
  -w "Time: %{time_total}s\n"
```

## 📈 Monitoring

### Built-in Metrics

- Token generation rate
- Response latency
- Memory usage
- Cache hit rate
- Error rates

### Custom Monitoring

```python
# Add to main.py
import psutil
import GPUtil

@app.get("/metrics")
async def metrics():
    gpu = GPUtil.getGPUs()[0]
    return {
        "gpu_memory_used": gpu.memoryUsed,
        "gpu_memory_total": gpu.memoryTotal,
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent
    }
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. Note that Llama-2 models have their own license terms from Meta.

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient inference
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Vue.js](https://vuejs.org/) for the frontend
- [Meta](https://meta.ai/) for the Llama-2 models

---

**Need help?** Open an issue or check the troubleshooting section above.
