"""
Example usage of the get_best_api_key_for_model function.

This script demonstrates how to use the new function to get the best
API key for a specific model from the available models list.
"""

from api_tracker_utils.tracker import get_best_api_key_for_model, track_model_call
from api_tracker_utils.config import AVAILABLE_MODELS


def example_usage():
    """Example showing how to get the best API key for different models."""
    
    print("=" * 60)
    print("EXAMPLE: Getting Best API Key for Specific Models")
    print("=" * 60)
    
    print(f"\nAvailable models: {AVAILABLE_MODELS}\n")
    
    # Example 1: Get best API key for gemini-2.5-flash
    try:
        model = "gemini-2.5-flash"
        print(f"Getting best API key for '{model}'...")
        api_key = get_best_api_key_for_model(model)
        print(f"✅ Best API key for {model}: ...{api_key[-6:]}\n")
        
        # Track the call (simulating actual usage)
        track_model_call(api_key, model)
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Example 2: Get best API key for gemini-2.5-flash-lite
    try:
        model = "gemini-2.5-flash-lite"
        print(f"Getting best API key for '{model}'...")
        api_key = get_best_api_key_for_model(model)
        print(f"✅ Best API key for {model}: ...{api_key[-6:]}\n")
        
        # Track the call (simulating actual usage)
        track_model_call(api_key, model)
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Example 3: Get best API key for gemma-3-27b-it
    try:
        model = "gemma-3-27b-it"
        print(f"Getting best API key for '{model}'...")
        api_key = get_best_api_key_for_model(model)
        print(f"✅ Best API key for {model}: ...{api_key[-6:]}\n")
        
        # Track the call (simulating actual usage)
        track_model_call(api_key, model)
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Example 4: Try an invalid model (will raise ValueError)
    try:
        model = "invalid-model"
        print(f"Getting best API key for '{model}'...")
        api_key = get_best_api_key_for_model(model)
        print(f"✅ Best API key for {model}: ...{api_key[-6:]}\n")
        
    except ValueError as e:
        print(f"❌ ValueError (expected): {e}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
