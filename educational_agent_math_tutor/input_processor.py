"""
Input processor for detecting and handling multimodal student inputs.
Supports text, image file paths, and base64 encoded images.
"""

import os
from typing import Dict, Any
from educational_agent_math_tutor.ocr_handler import (
    process_image_from_path,
    process_image_from_base64
)


def is_image_path(content: str) -> bool:
    """
    Check if content string is a file path to an image.
    
    Args:
        content: String to check
        
    Returns:
        True if it's an image file path
    """
    if not content or not isinstance(content, str):
        return False
    
    # Check for image extensions
    lower_content = content.lower()
    if lower_content.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
        return True
    
    # Check if file exists with image extension
    if os.path.exists(content):
        ext = os.path.splitext(content)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
            return True
    
    return False


def is_base64_image(content: str) -> bool:
    """
    Check if content string is a base64 encoded image.
    
    Args:
        content: String to check (or dict for LangGraph Studio format)
        
    Returns:
        True if it's a base64 image data URI
    """
    print(f"Checking if content is base64 image...")
    
    if not content:
        return False
    
    # Check for Streamlit format: direct data URI string
    if isinstance(content, str):
        if content.startswith('data:image/'):
            print("✅ Detected Streamlit base64 format")
            return True
        return False
    
    # Check for LangGraph Studio format: dict with image_url
    if isinstance(content, list) and len(content) > 0:
        if isinstance(content[0], dict) and 'image_url' in content[0]:
            url = content[0].get('image_url', {}).get('url', '')
            if url.startswith('data:image/'):
                print("✅ Detected LangGraph Studio base64 format")
                return True
    
    return False


def detect_and_process_input(content: str) -> Dict[str, Any]:
    """
    Detect input type and process accordingly.
    
    Handles:
    - Regular text (pass through)
    - Image file paths (run OCR)
    - Base64 encoded images (run OCR)
    
    Args:
        content: Input string (text, file path, or base64)
        
    Returns:
        Dict with:
            - processed_text: The final text to use
            - input_type: "text", "image_path", or "image_base64"
            - success: Whether processing succeeded
            - original_content: The original input (for debugging)
    """

    print("Reached Here - detect_and_process_input")

    if not content:
        return {
            "processed_text": "",
            "input_type": "empty",
            "success": True,
            "original_content": content
        }
    
    # Check for base64 image (must check before file path)
    if is_base64_image(content):
        print(f"📸 Detected base64 image input")
        
        # Extract base64 data based on format
        if isinstance(content, str):
            # Streamlit format: direct data URI
            base64_data = content
        else:
            # LangGraph Studio format: dict with image_url
            base64_data = content[0]['image_url']['url']
        
        result = process_image_from_base64(base64_data)
        return {
            "processed_text": result["text"],
            "input_type": "image_base64",
            "success": result["success"],
            "original_content": base64_data[:100] + "..." if len(base64_data) > 100 else base64_data
        }
    
    print(f"Content is not base64 image.")
    
    # Check for image file path
    if is_image_path(content):
        print(f"📸 Detected image file path: {content}")
        result = process_image_from_path(content)
        return {
            "processed_text": result["text"],
            "input_type": "image_path",
            "success": result["success"],
            "original_content": content
        }
    
    print(f"Content is not image file path.")
    
    # Regular text - pass through
    return {
        "processed_text": content,
        "input_type": "text",
        "success": True,
        "original_content": content
    }
