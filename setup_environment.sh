#!/bin/bash
# Setup script for local LLM inference environment

set -e  # Exit on any error

echo "🚀 Setting up Local LLM Inference Environment"
echo "=" * 50

# Check if running on Ubuntu/Debian
if command -v apt-get &> /dev/null; then
    echo "📦 Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        gcc-12 \
        g++-12 \
        git \
        wget \
        curl \
        python3-dev \
        python3-pip \
        python3-venv \
        redis-server \
        nvidia-cuda-toolkit
    echo "✅ System dependencies installed"
elif command -v yum &> /dev/null; then
    echo "📦 Installing system dependencies (RHEL/CentOS)..."
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        cmake \
        git \
        wget \
        curl \
        python3-devel \
        python3-pip \
        redis
    echo "✅ System dependencies installed"
else
    echo "⚠️  Please install the following manually:"
    echo "   - build-essential (gcc, g++, make)"
    echo "   - cmake"
    echo "   - python3-dev"
    echo "   - redis-server"
    echo "   - nvidia-cuda-toolkit (if using CUDA)"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Python version: $PYTHON_VERSION"

# Check if Python version is compatible
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "⚠️  Python 3.13 detected. Some packages may have compatibility issues."
    echo "   Consider using Python 3.11 or 3.12 for better compatibility."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Setup cancelled. Please install Python 3.11 or 3.12"
        exit 1
    fi
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set compiler environment variables for llama-cpp-python
echo "🔧 Setting compiler environment variables..."
export CC=gcc-12
export CXX=g++-12

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available. You may need to install CUDA drivers.')
"

# Test llama-cpp-python installation
echo "🧪 Testing llama-cpp-python installation..."
python3 -c "
try:
    import llama_cpp
    print('✅ llama-cpp-python installed successfully')
except ImportError as e:
    print(f'❌ llama-cpp-python installation failed: {e}')
    print('Try installing with: pip install llama-cpp-python --no-cache-dir')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Start Redis: redis-server --daemonize yes"
echo "2. Download model: python setup_model.py"
echo "3. Start API: python main.py"
echo "4. Test: python test_api.py"
echo ""
echo "If you encounter issues:"
echo "- Check CUDA installation: nvidia-smi"
echo "- Verify Redis is running: redis-cli ping"
echo "- Check model file exists: ls -la models/llama2-7b-q4.gguf"
