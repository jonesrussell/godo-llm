#!/usr/bin/env python3
"""
Simple script to create a .env file with default values.
"""

import os
from pathlib import Path

def create_env_file():
    """Create a .env file with default values"""
    env_file = Path(".env")
    
    # Default environment variables
    env_content = """# Hugging Face Configuration
HUGGINGFACE_HUB_TOKEN=your_token_here

# Model Configuration
MODEL_PATH=models/llama2-7b-q4.gguf
GPU_LAYERS=20
MODEL_CONTEXT_SIZE=2048
MODEL_BATCH_SIZE=512
MODELS_DIR=models

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=redis://localhost:6379/0

# Performance Configuration
MAX_TOKENS_DEFAULT=256
TEMPERATURE_DEFAULT=0.7
TOP_P_DEFAULT=0.9
TOP_K_DEFAULT=40

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# Development Configuration
DEBUG=false
RELOAD=false
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env file with default values!")
        print(f"📁 Location: {env_file.absolute()}")
        print("")
        print("🔧 Next steps:")
        print("1. Edit .env file and set your HUGGINGFACE_HUB_TOKEN")
        print("2. Adjust other settings as needed")
        print("3. Run: python setup_model.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

if __name__ == "__main__":
    create_env_file()
