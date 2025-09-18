#!/usr/bin/env python3
"""
Script to help set up environment variables for the LLM inference pipeline.
This creates a .env file with default values that users can customize.
"""

import os
from pathlib import Path

def create_env_file():
    """Create a .env file with default values"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled. Existing .env file preserved.")
            return False
    
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

def show_env_help():
    """Show help for environment variables"""
    print("🔧 Environment Variables Help")
    print("=" * 40)
    print("")
    print("📝 Required Variables:")
    print("  HUGGINGFACE_HUB_TOKEN - Your Hugging Face API token")
    print("")
    print("🤖 Model Configuration:")
    print("  MODEL_PATH - Path to your GGUF model file")
    print("  GPU_LAYERS - Number of layers to run on GPU (default: 20)")
    print("  MODEL_CONTEXT_SIZE - Context window size (default: 2048)")
    print("  MODEL_BATCH_SIZE - Batch size for processing (default: 512)")
    print("  MODELS_DIR - Directory for model files (default: models)")
    print("")
    print("🌐 API Configuration:")
    print("  API_HOST - Host to bind the API to (default: 0.0.0.0)")
    print("  API_PORT - Port to run the API on (default: 8000)")
    print("  API_WORKERS - Number of worker processes (default: 1)")
    print("")
    print("🗄️  Redis Configuration:")
    print("  REDIS_HOST - Redis server host (default: localhost)")
    print("  REDIS_PORT - Redis server port (default: 6379)")
    print("  REDIS_DB - Redis database number (default: 0)")
    print("")
    print("⚡ Performance Configuration:")
    print("  MAX_TOKENS_DEFAULT - Default max tokens (default: 256)")
    print("  TEMPERATURE_DEFAULT - Default temperature (default: 0.7)")
    print("  TOP_P_DEFAULT - Default top_p (default: 0.9)")
    print("  TOP_K_DEFAULT - Default top_k (default: 40)")
    print("")
    print("🔍 Development Configuration:")
    print("  DEBUG - Enable debug mode (default: false)")
    print("  RELOAD - Enable auto-reload (default: false)")

def main():
    """Main setup function"""
    print("🔧 Environment Setup for LLM Inference Pipeline")
    print("=" * 50)
    print("")
    
    while True:
        print("Choose an option:")
        print("1. Create .env file with default values")
        print("2. Show environment variables help")
        print("3. Exit")
        print("")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            create_env_file()
            break
        elif choice == "2":
            show_env_help()
            print("")
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
