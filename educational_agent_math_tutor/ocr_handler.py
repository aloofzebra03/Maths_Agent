"""
OCR Handler for processing images to text using Cloudflare Workers AI.
Supports both base64 data URIs and file paths.
"""

import base64
import tempfile
import os
from typing import Dict, Any
from cloudflare import Cloudflare

import dotenv
dotenv.load_dotenv(dotenv_path=".env", override=True)


# Cloudflare Configuration
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_OCR_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct"

# Initialize Cloudflare client

try:
    client = Cloudflare(api_token=CLOUDFLARE_API_TOKEN)
except Exception as e:
    raise Exception("Cloudflare API token not set.")

def encode_image_to_data_uri(file_path: str) -> str:
    """
    Encode a local image file to a base64 data URI.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Base64 encoded data URI string
    """
    mime_type = "image/jpeg"
    ext = file_path.lower().split('.')[-1]
    
    if ext == 'png':
        mime_type = "image/png"
    elif ext == 'webp':
        mime_type = "image/webp"
        
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{encoded_string}"


def process_image_from_path(image_path: str) -> Dict[str, Any]:
    """
    Process an image file through OCR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dict with extracted text and metadata
    """
    try:
        # Encode image to data URI
        image_data_uri = encode_image_to_data_uri(image_path)
        
        # Call OCR
        return _call_cloudflare_ocr(image_data_uri)
        
    except Exception as e:
        print(f"❌ OCR Error (file path): {e}")
        return {
            "text": "[Error: Could not read image. Please type your response instead.]",
            "input_type": "image",
            "success": False,
            "error": str(e)
        }


def process_image_from_base64(base64_string: str) -> Dict[str, Any]:
    """
    Process a base64 encoded image through OCR.
    Handles data URIs like "data:image/png;base64,..."
    
    Args:
        base64_string: Base64 encoded image data URI
        
    Returns:
        Dict with extracted text and metadata
    """
    try:
        # If it's already a data URI, use it directly
        if base64_string.startswith('data:image/'):
            return _call_cloudflare_ocr(base64_string)
        
        # Otherwise, decode and save to temp file, then process
        # Extract base64 data if it has prefix
        if ',' in base64_string:
            base64_data = base64_string.split(',', 1)[1]
        else:
            base64_data = base64_string
        
        # Decode to bytes
        image_bytes = base64.b64decode(base64_data)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        # Process from file
        result = process_image_from_path(tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return result
        
    except Exception as e:
        print(f"❌ OCR Error (base64): {e}")
        return {
            "text": "[Error: Could not decode image. Please type your response instead.]",
            "input_type": "image",
            "success": False,
            "error": str(e)
        }


def _call_cloudflare_ocr(image_data_uri: str) -> Dict[str, Any]:
    """
    Call Cloudflare Workers AI for OCR processing.
    
    Args:
        image_data_uri: Base64 data URI of the image
        
    Returns:
        Dict with extracted text and metadata
    """
    try:
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all text from this image. If it contains math problems, equations, or student work, transcribe exactly what you see. Output only the detected text."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri}
                }
            ]
        }]
        
        response = client.ai.run(
            model_name=CLOUDFLARE_OCR_MODEL,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            messages=messages
        )
        
        # Extract text from response
        text = ""
        if hasattr(response, 'result'):
            text = str(response.result)
        elif isinstance(response, dict):
            text = response.get('response', response.get('result', ''))
        
        if not text or text.strip() == "":
            text = "[No text detected in image. Please type your response instead.]"
        
        print(f"✅ OCR Success: Extracted {len(text)} characters")
        
        return {
            "text": text,
            "input_type": "image",
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Cloudflare OCR Error: {e}")
        return {
            "text": "[Error: OCR service failed. Please type your response instead.]",
            "input_type": "image",
            "success": False,
            "error": str(e)
        }
