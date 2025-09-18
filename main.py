import os
import json
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from llama_cpp import Llama
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Redis connection for caching (optional)
try:
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
    redis_client.ping()  # Test connection
    REDIS_AVAILABLE = True
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"⚠️  Redis not available: {e}")
    print("   Running without caching...")
    redis_client = None
    REDIS_AVAILABLE = False

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = int(os.getenv("MAX_TOKENS_DEFAULT", "256"))
    temperature: Optional[float] = float(os.getenv("TEMPERATURE_DEFAULT", "0.7"))
    top_p: Optional[float] = float(os.getenv("TOP_P_DEFAULT", "0.9"))
    top_k: Optional[int] = int(os.getenv("TOP_K_DEFAULT", "40"))
    stop: Optional[list] = None
    stream: Optional[bool] = True
    cache_key: Optional[str] = None

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time: float

# Global LLM instance
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global llm
    
    # Initialize LLM
    model_path = os.getenv("MODEL_PATH", "models/llama2-7b-q4.gguf")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run setup_model.py first.")
    
    # Configure GPU layers (adjust based on your VRAM)
    n_gpu_layers = int(os.getenv("GPU_LAYERS", "20"))
    n_ctx = int(os.getenv("MODEL_CONTEXT_SIZE", "2048"))
    n_batch = int(os.getenv("MODEL_BATCH_SIZE", "512"))
    
    print(f"Loading model from {model_path} with {n_gpu_layers} GPU layers...")
    print(f"Context size: {n_ctx}, Batch size: {n_batch}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,  # Context window
        n_batch=n_batch,  # Batch size for processing
        verbose=False
    )
    print("Model loaded successfully!")
    
    yield
    
    # Cleanup
    if llm:
        del llm

app = FastAPI(
    title="Local LLM Inference API",
    description="FastAPI service for local LLM inference with streaming support",
    version="1.0.0",
    lifespan=lifespan,
    debug=os.getenv("DEBUG", "false").lower() == "true"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_cache_key(prompt: str, params: Dict[str, Any]) -> str:
    """Generate cache key from prompt and parameters"""
    cache_data = {
        "prompt": prompt,
        "temperature": params.get("temperature", 0.7),
        "top_p": params.get("top_p", 0.9),
        "top_k": params.get("top_k", 40),
        "max_tokens": params.get("max_tokens", 256)
    }
    return f"llm_cache:{hash(json.dumps(cache_data, sort_keys=True))}"

@app.get("/")
async def root():
    return {"message": "Local LLM Inference API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "redis_connected": REDIS_AVAILABLE and redis_client.ping() if redis_client else False
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text with optional caching"""
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check cache first (if Redis is available)
    cache_key = request.cache_key or get_cache_key(request.prompt, request.dict())
    cached_result = None
    if REDIS_AVAILABLE and redis_client:
        try:
            cached_result = redis_client.get(cache_key)
        except Exception:
            pass
    
    if cached_result and not request.stream:
        return GenerateResponse(**json.loads(cached_result))
    
    # Generate new content
    try:
        if request.stream:
            return EventSourceResponse(
                stream_generation(request, cache_key),
                media_type="text/event-stream"
            )
        else:
            return await generate_non_streaming(request, cache_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def stream_generation(request: GenerateRequest, cache_key: str):
    """Stream generation with Server-Sent Events"""
    import time
    start_time = time.time()
    full_text = ""
    token_count = 0
    
    try:
        for token in llm(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop or [],
            echo=False,
            stream=True
        ):
            token_text = token["choices"][0]["text"]
            full_text += token_text
            token_count += 1
            
            # Send token as SSE event
            yield {
                "event": "token",
                "data": json.dumps({
                    "token": token_text,
                    "full_text": full_text,
                    "token_count": token_count
                })
            }
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
        
        # Send completion event
        generation_time = time.time() - start_time
        yield {
            "event": "complete",
            "data": json.dumps({
                "full_text": full_text,
                "token_count": token_count,
                "generation_time": generation_time
            })
        }
        
        # Cache the result (if Redis is available)
        if REDIS_AVAILABLE and redis_client:
            try:
                result = GenerateResponse(
                    text=full_text,
                    tokens_generated=token_count,
                    generation_time=generation_time
                )
                redis_client.setex(cache_key, 3600, result.json())  # Cache for 1 hour
            except Exception:
                pass
        
    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)})
        }

async def generate_non_streaming(request: GenerateRequest, cache_key: str):
    """Generate text without streaming"""
    import time
    start_time = time.time()
    
    result = llm(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop or [],
        echo=False,
        stream=False
    )
    
    generation_time = time.time() - start_time
    text = result["choices"][0]["text"]
    token_count = len(text.split())  # Approximate token count
    
    response = GenerateResponse(
        text=text,
        tokens_generated=token_count,
        generation_time=generation_time
    )
    
    # Cache the result (if Redis is available)
    if REDIS_AVAILABLE and redis_client:
        try:
            redis_client.setex(cache_key, 3600, response.json())
        except Exception:
            pass
    
    return response

@app.get("/models/info")
async def model_info():
    """Get information about the loaded model"""
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": llm.model_path,
        "context_size": llm.n_ctx(),
        "gpu_layers": llm.n_gpu_layers(),
        "batch_size": llm.n_batch()
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload
    )
