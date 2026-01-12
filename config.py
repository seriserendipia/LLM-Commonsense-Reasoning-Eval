"""
===========================================
Configuration File
===========================================
Configure all necessary settings in this file.
"""

import os
import sys

# ============================================================================
# 1. OpenRouter API Key Configuration
# ============================================================================
# Get your API key from https://openrouter.ai/settings/keys
OPENROUTER_API_KEY = "YOUR_API_KEY_HERE"

# Example:
# OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# ============================================================================
# 2. Dataset Path Configuration
# ============================================================================
# Set the path to your nlp-reasoning-benchmarks dataset folder.

# Option A (Recommended): Use a relative path.
# This assumes the dataset folder is in the same directory as the project.
DATASET_BASE_PATH = os.path.join(os.path.dirname(__file__), "nlp-reasoning-benchmarks")

# Option B: Use an absolute path.
# Use this if your dataset folder is in another location.
# DATASET_BASE_PATH = r"C:\Your\Path\To\nlp-reasoning-benchmarks"

# Examples:
# Windows:
# DATASET_BASE_PATH = r"D:\Datasets\nlp-reasoning-benchmarks"
# DATASET_BASE_PATH = r"E:\Research\Data\nlp-reasoning-benchmarks"
#
# Linux/Mac:
# DATASET_BASE_PATH = "/home/user/datasets/nlp-reasoning-benchmarks"
# DATASET_BASE_PATH = "/Users/username/datasets/nlp-reasoning-benchmarks"


# ============================================================================
# 3. Validate Configuration
# ============================================================================

def validate_config():
    """Validate that the configuration is correct."""
    errors = []
    
    # Check API key
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_API_KEY_HERE":
        errors.append(
            "❌ API Key not configured!\n"
            "   Please set OPENROUTER_API_KEY in config.py.\n"
            "   Get your key from https://openrouter.ai/settings/keys.\n"
        )
    
    # Check dataset path
    if not os.path.exists(DATASET_BASE_PATH):
        errors.append(
            f"❌ Dataset path does not exist: {DATASET_BASE_PATH}\n"
            f"   Please set the correct DATASET_BASE_PATH in config.py,\n"
            f"   or place the 'nlp-reasoning-benchmarks' folder in the project directory.\n"
        )
    
    if errors:
        print("\n" + "="*80)
        print("Configuration Errors:")
        print("="*80)
        for error in errors:
            print(error)
        print("="*80)
        print("Please fix the errors above and try again.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ Configuration validated successfully!")
    print("="*80)
    print(f"   API Key:      Loaded (ends with ...{OPENROUTER_API_KEY[-4:]})")
    print(f"   Dataset Path: {DATASET_BASE_PATH}")
    print("="*80)
    return True


if __name__ == "__main__":
    # Test the configuration
    validate_config()
