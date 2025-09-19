#!/usr/bin/env python3
"""
Script to help set up environment variables and system dependencies for the LLM inference pipeline.
This creates a .env file with default values and installs required system packages.
"""

import os
import subprocess
import sys
import platform
from pathlib import Path
from typing import List, Tuple

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

def install_system_dependencies() -> bool:
    """Install system dependencies based on the operating system"""
    system = platform.system().lower()
    
    if system == "linux":
        return _install_linux_dependencies()
    elif system == "darwin":  # macOS
        return _install_macos_dependencies()
    else:
        print(f"⚠️  Unsupported operating system: {system}")
        print("Please install dependencies manually:")
        print("- build-essential (gcc, g++, make)")
        print("- cmake")
        print("- python3-dev")
        print("- redis-server")
        return False

def _install_linux_dependencies() -> bool:
    """Install dependencies on Linux systems"""
    print("🐧 Installing Linux dependencies...")
    
    # Check if apt-get is available (Ubuntu/Debian)
    if subprocess.run(["which", "apt-get"], capture_output=True).returncode == 0:
        return _install_apt_dependencies()
    # Check if yum is available (RHEL/CentOS)
    elif subprocess.run(["which", "yum"], capture_output=True).returncode == 0:
        return _install_yum_dependencies()
    else:
        print("❌ Package manager not found. Please install dependencies manually.")
        return False

def _install_apt_dependencies() -> bool:
    """Install dependencies using apt-get"""
    packages = [
        "build-essential", "cmake", "gcc-12", "g++-12", "git", "wget", "curl",
        "python3-dev", "python3-pip", "python3-venv", "redis-server"
    ]
    
    try:
        print("📦 Updating package list...")
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        
        print("📦 Installing packages...")
        subprocess.run(["sudo", "apt-get", "install", "-y"] + packages, check=True)
        
        print("✅ Linux dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def _install_yum_dependencies() -> bool:
    """Install dependencies using yum"""
    packages = [
        "cmake", "git", "wget", "curl", "python3-devel", "python3-pip", "redis"
    ]
    
    try:
        print("📦 Installing development tools...")
        subprocess.run(["sudo", "yum", "groupinstall", "-y", "Development Tools"], check=True)
        
        print("📦 Installing packages...")
        subprocess.run(["sudo", "yum", "install", "-y"] + packages, check=True)
        
        print("✅ Linux dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def _install_macos_dependencies() -> bool:
    """Install dependencies on macOS"""
    print("🍎 macOS detected. Please install dependencies manually:")
    print("1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
    print("2. Install packages: brew install cmake gcc redis")
    return False

def validate_environment() -> bool:
    """Validate that required tools are available"""
    print("🔍 Validating environment...")
    
    required_commands = ["python3", "pip", "cmake", "gcc-12", "g++-12"]
    missing_commands = []
    
    for cmd in required_commands:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            missing_commands.append(cmd)
    
    if missing_commands:
        print(f"❌ Missing required commands: {', '.join(missing_commands)}")
        return False
    
    print("✅ All required commands are available!")
    return True

def main():
    """Main setup function"""
    print("🔧 Environment Setup for LLM Inference Pipeline")
    print("=" * 50)
    print("")
    
    while True:
        print("Choose an option:")
        print("1. Install system dependencies")
        print("2. Create .env file with default values")
        print("3. Validate environment")
        print("4. Show environment variables help")
        print("5. Exit")
        print("")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            install_system_dependencies()
            print("")
        elif choice == "2":
            create_env_file()
            break
        elif choice == "3":
            validate_environment()
            print("")
        elif choice == "4":
            show_env_help()
            print("")
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
