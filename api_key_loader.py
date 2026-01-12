"""
API Key Loader for OpenRouter
Loads the OpenRouter API key from config.py.
"""

import os
import sys


def load_openrouter_api_key():
    """
    Loads the OpenRouter API key from the config.py file.
    
    Returns:
        str: The OpenRouter API key.
        
    Raises:
        ImportError: If config.py cannot be found.
        ValueError: If the API key is not configured or is invalid.
    """
    try:
        # Import the configuration file
        import config
        
        # Get the API key
        api_key = config.OPENROUTER_API_KEY
        
        # Validate the API key
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            error_msg = (
                f"\n❌ API Key not configured!\n\n"
                f"Please set the OPENROUTER_API_KEY in the config.py file.\n"
                f"Get your API key from https://openrouter.ai/settings/keys.\n\n"
                f"Example:\n"
                f"OPENROUTER_API_KEY = \"sk-or-v1-xxxxxxxxxxxxx\"\n"
            )
            raise ValueError(error_msg)
        
        return api_key
        
    except ImportError:
        error_msg = (
            f"\n❌ Could not find the config.py file!\n\n"
            f"Please ensure that config.py exists in the project directory.\n"
            f"If not, create the file and set the OPENROUTER_API_KEY.\n"
        )
        raise ImportError(error_msg)


if __name__ == "__main__":
    # Test the loader
    try:
        key = load_openrouter_api_key()
        print(f"✅ Successfully loaded API key: {key[:8]}...{key[-4:]}")
    except Exception as e:
        print(f"❌ Error: {e}")
