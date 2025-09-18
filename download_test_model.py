#!/usr/bin/env python3
"""
Download a smaller, non-gated model for testing purposes.
This downloads TinyLlama-1.1B which is much smaller and doesn't require license approval.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Load environment variables
load_dotenv()

def download_tinyllama():
    """Download TinyLlama-1.1B for testing"""
    print("🚀 Downloading TinyLlama-1.1B for testing...")
    print("=" * 50)
    
    models_dir = Path(os.getenv("MODELS_DIR", "models"))
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Download TinyLlama (no license required)
        print("📥 Downloading TinyLlama-1.1B-Chat-v1.0...")
        snapshot_download(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            cache_dir=str(models_dir / "tinyllama"),
            local_files_only=False
        )
        print("✅ TinyLlama downloaded successfully!")
        print(f"📁 Model saved to: {models_dir / 'tinyllama'}")
        
        # Create a symlink for easy access
        model_path = models_dir / "tinyllama-1.1b.gguf"
        if not model_path.exists():
            print("⚠️  Note: This downloads the HuggingFace format, not GGUF.")
            print("   For GGUF format, you'll need to convert it or use a different model.")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def download_gguf_model():
    """Download a pre-converted GGUF model"""
    print("\n🔄 Trying to download a pre-converted GGUF model...")
    
    try:
        # Try to download a GGUF model from TheBloke
        print("📥 Downloading Llama-2-7B-Chat-GGUF...")
        models_dir = Path(os.getenv("MODELS_DIR", "models"))
        snapshot_download(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            cache_dir=str(models_dir / "llama2-7b-gguf"),
            local_files_only=False,
            allow_patterns="*.gguf"
        )
        print("✅ GGUF model downloaded successfully!")
        print("📁 Model saved to: models/llama2-7b-gguf")
        return True
        
    except Exception as e:
        print(f"❌ GGUF download failed: {e}")
        return False

def main():
    """Main download process"""
    print("🧪 Downloading Test Model")
    print("=" * 30)
    
    # Try GGUF model first (better for llama-cpp-python)
    if download_gguf_model():
        print("\n🎉 Success! You can now test the API with:")
        print("   python main.py")
        return
    
    # Fallback to TinyLlama
    if download_tinyllama():
        print("\n⚠️  TinyLlama downloaded, but you'll need to convert to GGUF format.")
        print("   For now, you can test the API structure without a model.")
    
    print("\n📝 Next steps:")
    print("1. Accept Llama 2 license at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
    print("2. Run: python setup_model.py")
    print("3. Or use the downloaded model for testing")

if __name__ == "__main__":
    main()
