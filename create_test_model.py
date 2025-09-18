#!/usr/bin/env python3
"""
Create a minimal test model for development/testing purposes.
This creates a simple text file that can be used to test the API structure.
"""

import os
from pathlib import Path

def create_test_model():
    """Create a minimal test model file"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create a simple text file as a placeholder
    test_model_path = models_dir / "test-model.txt"
    
    with open(test_model_path, "w") as f:
        f.write("This is a test model file for development.\n")
        f.write("Replace this with a real GGUF model file.\n")
        f.write("Download from: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf\n")
    
    print(f"✅ Created test model file: {test_model_path}")
    print("⚠️  This is just a placeholder. You need a real GGUF model file.")
    print("   Run 'python setup_model.py' to download the actual model.")

if __name__ == "__main__":
    create_test_model()
