#!/usr/bin/env python3
"""
Script to help set up Hugging Face authentication.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

def setup_hf_token():
    """Setup Hugging Face token"""
    print("🔐 Hugging Face Token Setup")
    print("=" * 40)
    
    # Check if token is already set
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        print(f"✅ Token already set: {token[:10]}...")
        try:
            login(token=token)
            print("✅ Successfully authenticated with Hugging Face!")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    print("📝 To get your token:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'read' permissions")
    print("3. Copy the token")
    print("")
    print("🔧 Then run one of these commands:")
    print("")
    print("Option 1 - Environment variable:")
    print("export HUGGINGFACE_HUB_TOKEN='your_token_here'")
    print("")
    print("Option 2 - Create .env file:")
    print("echo 'HUGGINGFACE_HUB_TOKEN=your_token_here' > .env")
    print("")
    print("Option 3 - Direct login (paste your token when prompted):")
    print("python -c \"from huggingface_hub import login; login()\"")
    
    return False

if __name__ == "__main__":
    setup_hf_token()
