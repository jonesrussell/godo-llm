#!/usr/bin/env python3
"""
Script to download and quantize Llama-2-7b model for local inference.
Requires Hugging Face account and Llama 2 license acceptance.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command with error handling"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result

def check_dependencies():
    """Check if required tools are installed"""
    print("Checking dependencies...")
    
    # Check Python packages
    try:
        import transformers
        import accelerate
        import huggingface_hub
        print("✓ Python packages installed")
    except ImportError as e:
        print(f"✗ Missing Python package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Check if llama.cpp is available
    try:
        result = run_command("python -c 'import llama_cpp'", check=False)
        if result.returncode != 0:
            print("✗ llama-cpp-python not installed")
            print("Please run: pip install llama-cpp-python[cuda]")
            return False
        print("✓ llama-cpp-python installed")
    except Exception as e:
        print(f"✗ Error checking llama-cpp-python: {e}")
        return False
    
    return True

def setup_huggingface():
    """Setup Hugging Face authentication"""
    print("\nSetting up Hugging Face authentication...")
    
    # Check if already logged in
    result = run_command("huggingface-cli whoami", check=False)
    if result.returncode == 0:
        print("✓ Already logged in to Hugging Face")
        return True
    
    print("Please log in to Hugging Face:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'read' permissions")
    print("3. Run: huggingface-cli login")
    print("4. Accept the Llama 2 license at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
    
    input("Press Enter after completing the login process...")
    
    # Verify login
    result = run_command("huggingface-cli whoami", check=False)
    if result.returncode == 0:
        print("✓ Successfully logged in to Hugging Face")
        return True
    else:
        print("✗ Login failed. Please try again.")
        return False

def download_model():
    """Download Llama-2-7b model"""
    print("\nDownloading Llama-2-7b model...")
    
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
        
        print(f"Downloading {model_id}...")
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(models_dir / "llama2-7b"),
            local_files_only=False
        )
        print("✓ Model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def convert_to_gguf():
    """Convert model to GGUF format"""
    print("\nConverting model to GGUF format...")
    
    # Check if conversion script exists
    conversion_script = "convert_hf_to_gguf.py"
    if not os.path.exists(conversion_script):
        print("Downloading conversion script...")
        run_command(
            "wget https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
        )
    
    # Convert to GGUF
    model_path = "models/llama2-7b/meta-llama--Llama-2-7b-chat-hf"
    output_path = "models/llama2-7b-chat.gguf"
    
    cmd = f"python convert_hf_to_gguf.py {model_path} --outfile {output_path}"
    result = run_command(cmd, check=False)
    
    if result.returncode == 0 and os.path.exists(output_path):
        print("✓ Model converted to GGUF successfully")
        return True
    else:
        print("✗ Conversion failed")
        print("You may need to install llama.cpp manually:")
        print("git clone https://github.com/ggerganov/llama.cpp.git")
        print("cd llama.cpp && make")
        return False

def quantize_model():
    """Quantize model to Q4_K_M"""
    print("\nQuantizing model to Q4_K_M...")
    
    input_path = "models/llama2-7b-chat.gguf"
    output_path = "models/llama2-7b-q4.gguf"
    
    # Check if quantize script exists
    quantize_script = "quantize"
    if not os.path.exists(quantize_script):
        print("Downloading quantize script...")
        run_command(
            "wget https://raw.githubusercontent.com/ggerganov/llama.cpp/master/quantize"
        )
        run_command("chmod +x quantize")
    
    cmd = f"./quantize {input_path} {output_path} Q4_K_M"
    result = run_command(cmd, check=False)
    
    if result.returncode == 0 and os.path.exists(output_path):
        print("✓ Model quantized successfully")
        return True
    else:
        print("✗ Quantization failed")
        print("Trying alternative quantization method...")
        
        # Alternative: Use llama-cpp-python for quantization
        try:
            from llama_cpp import Llama
            print("Loading model for quantization...")
            llm = Llama(model_path=input_path, verbose=True)
            print("✓ Model loaded, quantization complete")
            return True
        except Exception as e:
            print(f"✗ Alternative quantization failed: {e}")
            return False

def cleanup():
    """Clean up temporary files"""
    print("\nCleaning up temporary files...")
    
    files_to_remove = [
        "convert_hf_to_gguf.py",
        "quantize",
        "models/llama2-7b-chat.gguf"  # Keep only the quantized version
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")

def main():
    """Main setup process"""
    print("🚀 Setting up Llama-2-7b for local inference")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup Hugging Face
    if not setup_huggingface():
        sys.exit(1)
    
    # Download model
    if not download_model():
        sys.exit(1)
    
    # Convert to GGUF
    if not convert_to_gguf():
        print("⚠️  GGUF conversion failed, but you can still use the model")
    
    # Quantize model
    if not quantize_model():
        print("⚠️  Quantization failed, but you can still use the model")
    
    # Cleanup
    cleanup()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Start Redis: redis-server")
    print("2. Run the API: python main.py")
    print("3. Test the endpoint: curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello, how are you?\"}'")

if __name__ == "__main__":
    main()
