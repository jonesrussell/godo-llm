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
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from enum import Enum

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

# Hugging Face imports
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TextIteratorStreamer, pipeline
)
from threading import Thread
from queue import Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enums
class ModelProvider(str, Enum):
    """Available model providers."""
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


# Configuration
class Config:
    """Application configuration."""
    
    # Model provider settings
    DEFAULT_PROVIDER: ModelProvider = ModelProvider(os.getenv("DEFAULT_PROVIDER", "local"))
    HUGGINGFACE_MODEL_NAME: str = os.getenv("HUGGINGFACE_MODEL_NAME", "microsoft/DialoGPT-medium")
    HUGGINGFACE_CACHE_DIR: str = os.getenv("HUGGINGFACE_CACHE_DIR", "./hf_cache")
    
    # Local model settings - will be auto-detected
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
    provider: Optional[ModelProvider] = Field(
        default=None,
        description="Model provider to use (overrides default)"
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
    current_provider: ModelProvider
    local_model_loaded: bool
    huggingface_model_loaded: bool
    redis_connected: bool
    uptime: float


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    current_provider: ModelProvider
    available_providers: List[ModelProvider]
    
    # Local model info
    local_model_path: Optional[str] = None
    context_size: Optional[int] = None
    gpu_layers: Optional[int] = None
    batch_size: Optional[int] = None
    local_model_loaded: bool = False
    
    # Hugging Face model info
    huggingface_model_name: Optional[str] = None
    huggingface_model_loaded: bool = False
    device: Optional[str] = None
    torch_dtype: Optional[str] = None


class SwitchProviderRequest(BaseModel):
    """Request model for switching model providers."""
    
    provider: ModelProvider = Field(..., description="Provider to switch to")
    huggingface_model_name: Optional[str] = Field(
        default=None,
        description="Hugging Face model name (only for HF provider)"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str
    error_code: Optional[str] = None
    timestamp: float


# Global state
class AppState:
    """Application state container."""
    
    def __init__(self):
        self.local_llm: Optional[Llama] = None
        self.hf_model: Optional[Any] = None
        self.hf_tokenizer: Optional[Any] = None
        self.current_provider: ModelProvider = config.DEFAULT_PROVIDER
        self.redis_client: Optional[redis.Redis] = None
        self.start_time: float = time.time()
        self.redis_available: bool = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time
    
    @property
    def active_model(self) -> Optional[Union[Llama, Any]]:
        """Get the currently active model based on provider."""
        if self.current_provider == ModelProvider.LOCAL:
            return self.local_llm
        else:
            return self.hf_model


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
    def initialize_local_model() -> Llama:
        """Initialize the local LLM model."""
        # Auto-detect model if not specified
        if not config.MODEL_PATH:
            config.MODEL_PATH = LLMManager.find_available_model()
        
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {config.MODEL_PATH}. "
                "Please run setup_model.py first."
            )
        
        logger.info(f"Loading local model from {config.MODEL_PATH}")
        logger.info(f"GPU layers: {config.GPU_LAYERS}, Context: {config.MODEL_CONTEXT_SIZE}")
        
        try:
            llm = Llama(
                model_path=config.MODEL_PATH,
                n_gpu_layers=config.GPU_LAYERS,
                n_ctx=config.MODEL_CONTEXT_SIZE,
                n_batch=config.MODEL_BATCH_SIZE,
                verbose=False
            )
            logger.info("Local model loaded successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise


class HuggingFaceManager:
    """Manages Hugging Face model initialization and operations."""
    
    @staticmethod
    def initialize_hf_model(model_name: str = None) -> tuple[Any, Any]:
        """Initialize Hugging Face model and tokenizer."""
        model_name = model_name or config.HUGGINGFACE_MODEL_NAME
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Hugging Face model: {model_name}")
        logger.info(f"Device: {device}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=config.HUGGINGFACE_CACHE_DIR
            )
            
            # Fix tokenizer for DialoGPT models
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=config.HUGGINGFACE_CACHE_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                model = model.to(device)
            
            logger.info("Hugging Face model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
            raise
    
    @staticmethod
    def generate_hf_streaming(
        model: Any, 
        tokenizer: Any, 
        prompt: str, 
        max_tokens: int, 
        temperature: float, 
        top_p: float, 
        stop: List[str]
    ) -> AsyncGenerator[str, None]:
        """Generate text with Hugging Face model using streaming."""
        device = next(model.parameters()).device
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            tokenizer, 
            timeout=10.0, 
            skip_special_tokens=True
        )
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,  # Prevent repetitive outputs
            "no_repeat_ngram_size": 3,  # Prevent 3-gram repetition
            "streamer": streamer,
        }
        
        # Start generation in a separate thread
        generation_kwargs.update(inputs)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        try:
            for token in streamer:
                yield token
        except Exception as e:
            logger.error(f"Hugging Face streaming error: {e}")
            raise
        finally:
            thread.join()
    
    @staticmethod
    def generate_hf_non_streaming(
        model: Any, 
        tokenizer: Any, 
        prompt: str, 
        max_tokens: int, 
        temperature: float, 
        top_p: float, 
        stop: List[str]
    ) -> str:
        """Generate text with Hugging Face model without streaming."""
        device = next(model.parameters()).device
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,  # Prevent repetitive outputs
            "no_repeat_ngram_size": 3,  # Prevent 3-gram repetition
        }
        
        try:
            # Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"Hugging Face generation error: {e}")
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
        
        # Initialize local model (if available)
        try:
            app_state.local_llm = LLMManager.initialize_local_model()
            logger.info("Local model initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize local model: {e}")
            app_state.local_llm = None
        
        # Initialize Hugging Face model (if available)
        try:
            app_state.hf_model, app_state.hf_tokenizer = HuggingFaceManager.initialize_hf_model()
            logger.info("Hugging Face model initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Hugging Face model: {e}")
            app_state.hf_model = None
            app_state.hf_tokenizer = None
        
        # Validate that at least one model is available
        if not app_state.local_llm and not app_state.hf_model:
            raise RuntimeError("No models could be initialized. Please check your configuration.")
        
        # Set default provider based on availability
        if app_state.current_provider == ModelProvider.LOCAL and not app_state.local_llm:
            if app_state.hf_model:
                app_state.current_provider = ModelProvider.HUGGINGFACE
                logger.info("Switching to Hugging Face provider (local model unavailable)")
            else:
                raise RuntimeError("No models available for the configured provider")
        elif app_state.current_provider == ModelProvider.HUGGINGFACE and not app_state.hf_model:
            if app_state.local_llm:
                app_state.current_provider = ModelProvider.LOCAL
                logger.info("Switching to local provider (Hugging Face model unavailable)")
            else:
                raise RuntimeError("No models available for the configured provider")
        
        logger.info(f"Service startup completed successfully with provider: {app_state.current_provider}")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down service...")
    
    if app_state.redis_client:
        await RedisManager.close(app_state.redis_client)
    
    if app_state.local_llm:
        del app_state.local_llm
    
    if app_state.hf_model:
        del app_state.hf_model
    
    if app_state.hf_tokenizer:
        del app_state.hf_tokenizer
    
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


# Mount static files for the frontend
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

# Routes
@app.get("/vite.svg")
async def vite_svg():
    """Serve the vite.svg favicon."""
    try:
        with open("frontend/dist/vite.svg", "r") as f:
            from fastapi.responses import Response
            return Response(content=f.read(), media_type="image/svg+xml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="vite.svg not found")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Vue.js frontend."""
    try:
        with open("frontend/dist/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>Local LLM Inference API</h1>
                    <p>Status: Running</p>
                    <p>Version: 1.0.0</p>
                    <p>Frontend not found. Please run 'npm run build' in the frontend directory.</p>
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
    
    # Determine overall health status
    has_model = app_state.local_llm is not None or app_state.hf_model is not None
    status = "healthy" if has_model else "degraded"
    
    return HealthResponse(
        status=status,
        current_provider=app_state.current_provider,
        local_model_loaded=app_state.local_llm is not None,
        huggingface_model_loaded=app_state.hf_model is not None,
        redis_connected=redis_connected,
        uptime=app_state.uptime
    )


@app.get("/models/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if not app_state.local_llm and not app_state.hf_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded"
        )
    
    available_providers = []
    if app_state.local_llm:
        available_providers.append(ModelProvider.LOCAL)
    if app_state.hf_model:
        available_providers.append(ModelProvider.HUGGINGFACE)
    
    return ModelInfoResponse(
        current_provider=app_state.current_provider,
        available_providers=available_providers,
        local_model_path=config.MODEL_PATH if app_state.local_llm else None,
        context_size=config.MODEL_CONTEXT_SIZE if app_state.local_llm else None,
        gpu_layers=config.GPU_LAYERS if app_state.local_llm else None,
        batch_size=config.MODEL_BATCH_SIZE if app_state.local_llm else None,
        local_model_loaded=app_state.local_llm is not None,
        huggingface_model_name=config.HUGGINGFACE_MODEL_NAME if app_state.hf_model else None,
        huggingface_model_loaded=app_state.hf_model is not None,
        device=str(app_state.device) if app_state.hf_model else None,
        torch_dtype=str(app_state.hf_model.dtype) if app_state.hf_model else None
    )


@app.post("/models/switch-provider", response_model=ModelInfoResponse)
async def switch_provider(request: SwitchProviderRequest):
    """Switch between model providers at runtime."""
    # Validate provider availability
    if request.provider == ModelProvider.LOCAL and not app_state.local_llm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Local model not available"
        )
    
    if request.provider == ModelProvider.HUGGINGFACE and not app_state.hf_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hugging Face model not available"
        )
    
    # Switch provider
    old_provider = app_state.current_provider
    app_state.current_provider = request.provider
    
    # If switching to HF and a different model is requested, reload
    if (request.provider == ModelProvider.HUGGINGFACE and 
        request.huggingface_model_name and 
        request.huggingface_model_name != config.HUGGINGFACE_MODEL_NAME):
        
        try:
            logger.info(f"Loading new Hugging Face model: {request.huggingface_model_name}")
            
            # Clean up old model
            if app_state.hf_model:
                del app_state.hf_model
            if app_state.hf_tokenizer:
                del app_state.hf_tokenizer
            
            # Load new model
            app_state.hf_model, app_state.hf_tokenizer = HuggingFaceManager.initialize_hf_model(
                request.huggingface_model_name
            )
            config.HUGGINGFACE_MODEL_NAME = request.huggingface_model_name
            
        except Exception as e:
            # Revert on failure
            app_state.current_provider = old_provider
            logger.error(f"Failed to load new Hugging Face model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
    
    logger.info(f"Switched from {old_provider} to {request.provider} provider")
    
    # Return updated model info
    return await model_info()


@app.get("/models/providers")
async def get_available_providers():
    """Get list of available model providers."""
    providers = []
    
    if app_state.local_llm:
        providers.append({
            "name": ModelProvider.LOCAL,
            "display_name": "Local Model",
            "description": "Local GGUF model via llama-cpp-python",
            "model_path": config.MODEL_PATH,
            "context_size": config.MODEL_CONTEXT_SIZE,
            "gpu_layers": config.GPU_LAYERS
        })
    
    if app_state.hf_model:
        providers.append({
            "name": ModelProvider.HUGGINGFACE,
            "display_name": "Hugging Face",
            "description": "Hugging Face model via transformers",
            "model_name": config.HUGGINGFACE_MODEL_NAME,
            "device": str(app_state.device),
            "torch_dtype": str(app_state.hf_model.dtype)
        })
    
    return {
        "current_provider": app_state.current_provider,
        "available_providers": providers
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text with optional caching and streaming."""
    # Determine which provider to use
    provider = request.provider or app_state.current_provider
    
    # Validate provider availability
    if provider == ModelProvider.LOCAL and not app_state.local_llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local model not available"
        )
    
    if provider == ModelProvider.HUGGINGFACE and not app_state.hf_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hugging Face model not available"
        )
    
    # Generate cache key (include provider in cache key)
    request_dict = request.dict()
    request_dict["provider"] = provider
    cache_key = request.cache_key or CacheManager.generate_cache_key(
        request.prompt, request_dict
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
                stream_generation(request, cache_key, provider),
                media_type="text/event-stream"
            )
        else:
            return await generate_non_streaming(request, cache_key, provider)
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


async def stream_generation(
    request: GenerateRequest, 
    cache_key: str,
    provider: ModelProvider
) -> AsyncGenerator[Dict[str, str], None]:
    """Stream text generation with Server-Sent Events."""
    start_time = time.time()
    full_text = ""
    token_count = 0
    
    try:
        if provider == ModelProvider.LOCAL:
            # Local model streaming
            for token in app_state.local_llm(
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
        
        else:  # Hugging Face model streaming
            async for token_text in HuggingFaceManager.generate_hf_streaming(
                app_state.hf_model,
                app_state.hf_tokenizer,
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.stop or []
            ):
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
    cache_key: str,
    provider: ModelProvider
) -> GenerateResponse:
    """Generate text without streaming."""
    start_time = time.time()
    
    try:
        if provider == ModelProvider.LOCAL:
            # Local model non-streaming
            result = app_state.local_llm(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop or [],
                echo=False,
                stream=False
            )
            
            text = result["choices"][0]["text"]
            # Better token counting - this is approximate
            token_count = len(text.split())
        
        else:  # Hugging Face model non-streaming
            text = HuggingFaceManager.generate_hf_non_streaming(
                app_state.hf_model,
                app_state.hf_tokenizer,
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.stop or []
            )
            # Better token counting - this is approximate
            token_count = len(text.split())
        
        generation_time = time.time() - start_time
        
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
