"""
Local LLM Inference API

A FastAPI service for local LLM inference with streaming support, caching, and monitoring.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration."""
    
    # Model settings - will be auto-detected
    MODEL_PATH: str = os.getenv("MODEL_PATH", "")
    GPU_LAYERS: int = int(os.getenv("GPU_LAYERS", "20"))
    MODEL_CONTEXT_SIZE: int = int(os.getenv("MODEL_CONTEXT_SIZE", "2048"))
    MODEL_BATCH_SIZE: int = int(os.getenv("MODEL_BATCH_SIZE", "512"))
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_URL: str = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    
    # Generation defaults
    MAX_TOKENS_DEFAULT: int = int(os.getenv("MAX_TOKENS_DEFAULT", "256"))
    TEMPERATURE_DEFAULT: float = float(os.getenv("TEMPERATURE_DEFAULT", "0.7"))
    TOP_P_DEFAULT: float = float(os.getenv("TOP_P_DEFAULT", "0.9"))
    TOP_K_DEFAULT: int = int(os.getenv("TOP_K_DEFAULT", "40"))
    
    # Cache settings
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Streaming settings
    STREAM_DELAY: float = float(os.getenv("STREAM_DELAY", "0.01"))


config = Config()

# Pydantic models
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., min_length=1, max_length=4000, description="Input prompt")
    max_tokens: Optional[int] = Field(
        default=config.MAX_TOKENS_DEFAULT,
        ge=1,
        le=2048,
        description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=config.TEMPERATURE_DEFAULT,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=config.TOP_P_DEFAULT,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling"
    )
    top_k: Optional[int] = Field(
        default=config.TOP_K_DEFAULT,
        ge=1,
        le=100,
        description="Top-k sampling"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    stream: Optional[bool] = Field(
        default=True,
        description="Enable streaming response"
    )
    cache_key: Optional[str] = Field(
        default=None,
        description="Custom cache key"
    )
    
    @field_validator('stop')
    @classmethod
    def validate_stop_sequences(cls, v):
        if v is not None and len(v) > 10:
            raise ValueError("Maximum 10 stop sequences allowed")
        return v


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    
    text: str
    tokens_generated: int
    generation_time: float
    cached: bool = False


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    model_loaded: bool
    redis_connected: bool
    uptime: float


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_path: str
    context_size: int
    gpu_layers: int
    batch_size: int
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str
    error_code: Optional[str] = None
    timestamp: float


# Global state
class AppState:
    """Application state container."""
    
    def __init__(self):
        self.llm: Optional[Llama] = None
        self.redis_client: Optional[redis.Redis] = None
        self.start_time: float = time.time()
        self.redis_available: bool = False
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time


app_state = AppState()


# Redis connection manager
class RedisManager:
    """Manages Redis connections and operations."""
    
    @staticmethod
    async def initialize() -> Optional[redis.Redis]:
        """Initialize Redis connection."""
        try:
            client = redis.from_url(config.REDIS_URL, decode_responses=True)
            await client.ping()
            logger.info("Redis connected successfully")
            app_state.redis_available = True
            return client
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            logger.info("Running without caching...")
            app_state.redis_available = False
            return None
    
    @staticmethod
    async def close(client: Optional[redis.Redis]) -> None:
        """Close Redis connection."""
        if client:
            try:
                await client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")


# Cache utilities
class CacheManager:
    """Manages caching operations."""
    
    @staticmethod
    def generate_cache_key(prompt: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key."""
        cache_data = {
            "prompt": prompt,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),
            "max_tokens": params.get("max_tokens")
        }
        content = json.dumps(cache_data, sort_keys=True)
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    @staticmethod
    async def get_cached_result(key: str) -> Optional[GenerateResponse]:
        """Retrieve cached result."""
        if not app_state.redis_available or not app_state.redis_client:
            return None
        
        try:
            cached_data = await app_state.redis_client.get(key)
            if cached_data:
                data = json.loads(cached_data)
                data['cached'] = True
                return GenerateResponse(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    @staticmethod
    async def cache_result(key: str, result: GenerateResponse) -> None:
        """Cache generation result."""
        if not app_state.redis_available or not app_state.redis_client:
            return
        
        try:
            await app_state.redis_client.setex(
                key,
                config.CACHE_TTL,
                result.json(exclude={'cached'})
            )
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")


# LLM manager
class LLMManager:
    """Manages LLM initialization and operations."""
    
    @staticmethod
    def find_available_model() -> str:
        """Find the first available GGUF model."""
        models_dir = Path("models")
        if not models_dir.exists():
            raise FileNotFoundError("Models directory not found. Please run setup_model.py first.")
        
        # Look for GGUF models in order of preference
        model_candidates = [
            "models/llama2-7b-q4.gguf",
            "models/llama2-7b-chat.gguf", 
            "models/tinyllama-1.1b-chat.gguf",
            "models/tinyllama.gguf"
        ]
        
        # Also search for any .gguf files in the models directory
        for gguf_file in models_dir.rglob("*.gguf"):
            model_candidates.append(str(gguf_file))
        
        for model_path in model_candidates:
            if os.path.exists(model_path):
                logger.info(f"Found model: {model_path}")
                return model_path
        
        raise FileNotFoundError(
            "No GGUF model found. Please run setup_model.py to download and convert a model."
        )
    
    @staticmethod
    def initialize_model() -> Llama:
        """Initialize the LLM model."""
        # Auto-detect model if not specified
        if not config.MODEL_PATH:
            config.MODEL_PATH = LLMManager.find_available_model()
        
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {config.MODEL_PATH}. "
                "Please run setup_model.py first."
            )
        
        logger.info(f"Loading model from {config.MODEL_PATH}")
        logger.info(f"GPU layers: {config.GPU_LAYERS}, Context: {config.MODEL_CONTEXT_SIZE}")
        
        try:
            llm = Llama(
                model_path=config.MODEL_PATH,
                n_gpu_layers=config.GPU_LAYERS,
                n_ctx=config.MODEL_CONTEXT_SIZE,
                n_batch=config.MODEL_BATCH_SIZE,
                verbose=False
            )
            logger.info("Model loaded successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting LLM API service...")
    
    try:
        # Initialize Redis
        app_state.redis_client = await RedisManager.initialize()
        
        # Initialize LLM
        app_state.llm = LLMManager.initialize_model()
        
        logger.info("Service startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down service...")
    
    if app_state.redis_client:
        await RedisManager.close(app_state.redis_client)
    
    if app_state.llm:
        del app_state.llm
    
    logger.info("Service shutdown completed")


# FastAPI app
app = FastAPI(
    title="Local LLM Inference API",
    description="FastAPI service for local LLM inference with streaming support",
    version="1.0.0",
    lifespan=lifespan,
    debug=config.DEBUG,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=time.time()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=time.time()
        ).dict()
    )


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Vue.js frontend."""
    try:
        with open("frontend/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>Local LLM Inference API</h1>
                    <p>Status: Running</p>
                    <p>Version: 1.0.0</p>
                    <p>Frontend not found. Please check the frontend directory.</p>
                </body>
            </html>
            """,
            status_code=200
        )


@app.get("/api/info", response_model=Dict[str, str])
async def api_info():
    """API information endpoint."""
    return {
        "message": "Local LLM Inference API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    redis_connected = False
    
    if app_state.redis_available and app_state.redis_client:
        try:
            await app_state.redis_client.ping()
            redis_connected = True
        except Exception:
            redis_connected = False
    
    return HealthResponse(
        status="healthy" if app_state.llm else "degraded",
        model_loaded=app_state.llm is not None,
        redis_connected=redis_connected,
        uptime=app_state.uptime
    )


@app.get("/models/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if not app_state.llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfoResponse(
        model_path=config.MODEL_PATH,
        context_size=config.MODEL_CONTEXT_SIZE,
        gpu_layers=config.GPU_LAYERS,
        batch_size=config.MODEL_BATCH_SIZE,
        model_loaded=True
    )


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text with optional caching and streaming."""
    if not app_state.llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Generate cache key
    cache_key = request.cache_key or CacheManager.generate_cache_key(
        request.prompt, request.dict()
    )
    
    # Check cache for non-streaming requests
    if not request.stream:
        cached_result = await CacheManager.get_cached_result(cache_key)
        if cached_result:
            return cached_result
    
    # Generate content
    try:
        if request.stream:
            return EventSourceResponse(
                stream_generation(request, cache_key),
                media_type="text/event-stream"
            )
        else:
            return await generate_non_streaming(request, cache_key)
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


async def stream_generation(
    request: GenerateRequest, 
    cache_key: str
) -> AsyncGenerator[Dict[str, str], None]:
    """Stream text generation with Server-Sent Events."""
    start_time = time.time()
    full_text = ""
    token_count = 0
    
    try:
        for token in app_state.llm(
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
            
            # Send token event
            yield {
                "event": "token",
                "data": json.dumps({
                    "token": token_text,
                    "full_text": full_text,
                    "token_count": token_count
                })
            }
            
            # Prevent overwhelming the client
            await asyncio.sleep(config.STREAM_DELAY)
        
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
        
        # Cache the result
        result = GenerateResponse(
            text=full_text,
            tokens_generated=token_count,
            generation_time=generation_time
        )
        await CacheManager.cache_result(cache_key, result)
        
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)})
        }


async def generate_non_streaming(
    request: GenerateRequest, 
    cache_key: str
) -> GenerateResponse:
    """Generate text without streaming."""
    start_time = time.time()
    
    try:
        result = app_state.llm(
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
        # Better token counting - this is approximate
        token_count = len(text.split())
        
        response = GenerateResponse(
            text=text,
            tokens_generated=token_count,
            generation_time=generation_time
        )
        
        # Cache the result
        await CacheManager.cache_result(cache_key, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Non-streaming generation error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
        log_level="debug" if config.DEBUG else "info"
    )
