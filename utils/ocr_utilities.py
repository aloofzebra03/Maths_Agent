"""
OCR utilities for image-to-text extraction.

This module keeps OCR implementation separate from shared_utils so the core
helper module stays focused on generic LLM and prompt utilities.
"""

import base64
import json
import os
import tempfile
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from utils.shared_utils import extract_json_block, get_llm
from api_tracker_utils.error import APITrackerError

try:
    from api_tracker_utils.tracker import get_best_api_key_for_model, track_model_call
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    print("[OCR] API tracker unavailable; using environment key fallback.")


OCR_MODEL = "gemma-3-27b-it"
OCR_TEMPERATURE = 0.0
OCR_PROMPT = (
    "Transcribe the math/text in this image into LaTeX format. "
    "Do not explain steps, do not include headers. "
    "Give only the text written in the image and nothing else. "
    'Return your answer strictly as JSON: {"latex_text": "<transcription>"}'
)

_CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
_CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
_CLOUDFLARE_OCR_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct"

_cloudflare_client = None
if _CLOUDFLARE_API_TOKEN:
    try:
        from cloudflare import Cloudflare

        _cloudflare_client = Cloudflare(api_token=_CLOUDFLARE_API_TOKEN)
    except Exception as exc:
        print(f"[OCR] Failed to initialize Cloudflare client: {exc}")
else:
    print("[OCR] CLOUDFLARE_API_TOKEN not set; Cloudflare OCR disabled.")


def _get_ocr_llm():
    """Build the OCR model using the same LangChain Gemini path as other helpers."""
    if TRACKER_AVAILABLE:
        try:
            api_key = get_best_api_key_for_model(OCR_MODEL)
        except APITrackerError:
            raise
        except Exception as exc:
            raise APITrackerError(f"Tracker selection failed for model '{OCR_MODEL}': {exc}") from exc
        print(f"[OCR] Using tracked API key ...{api_key[-6:]} for model: {OCR_MODEL}")
        try:
            track_model_call(api_key, OCR_MODEL)
        except APITrackerError:
            raise
        except Exception as exc:
            raise APITrackerError(f"Tracker call-tracking failed for model '{OCR_MODEL}': {exc}") from exc
        return get_llm(api_key=api_key, model=OCR_MODEL, temperature=OCR_TEMPERATURE)

    api_key = os.getenv("GOOGLE_API_KEY_4")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY_4 is not set")

    print(f"[OCR] Using GOOGLE_API_KEY_4 for model: {OCR_MODEL}")
    return get_llm(api_key=api_key, model=OCR_MODEL, temperature=OCR_TEMPERATURE)


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def encode_image_to_data_uri(file_path: str) -> str:
    """Encode a local image file to a base64 data URI."""
    mime_type = "image/jpeg"
    ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""

    if ext == "png":
        mime_type = "image/png"
    elif ext == "webp":
        mime_type = "image/webp"
    elif ext == "gif":
        mime_type = "image/gif"

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_string}"


def _call_gemma_ocr_from_path(file_path: str) -> Dict[str, Any]:
    """Call the OCR Gemini model through LangChain using an image file."""
    image_data_uri = encode_image_to_data_uri(file_path)
    # Keep tracker/model selection errors visible to API entrypoints.
    llm = _get_ocr_llm()

    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ]
        )

        response = llm.invoke([message])
        response_text = _content_to_text(getattr(response, "content", response))

        try:
            json_str = extract_json_block(response_text)
            data = json.loads(json_str)
            text = data.get("latex_text", response_text)
        except Exception:
            text = response_text

        text = (text or "").strip()
        print(f"[OCR] Gemma OCR success: extracted {len(text)} characters")
        return {
            "text": text,
            "input_type": "image",
            "success": True,
            "model": OCR_MODEL,
        }
    except Exception as exc:
        print(f"[OCR] Gemma OCR error: {exc}")
        return {
            "text": "[Error: OCR fallback failed. Please type your response instead.]",
            "input_type": "image",
            "success": False,
            "error": str(exc),
        }


def _call_cloudflare_ocr(image_data_uri: str) -> Dict[str, Any]:
    """Call Cloudflare Workers AI for OCR processing."""
    try:
        if _cloudflare_client is None:
            return {
                "text": "[Error: Cloudflare OCR client is not configured.]",
                "input_type": "image",
                "success": False,
                "error": "CLOUDFLARE_API_TOKEN is missing or invalid",
            }

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all text from this image. If it contains math problems, equations, or student work, transcribe exactly what you see. Output only the detected text.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri},
                },
            ],
        }]

        response = _cloudflare_client.ai.run(
            model_name=_CLOUDFLARE_OCR_MODEL,
            account_id=_CLOUDFLARE_ACCOUNT_ID,
            messages=messages,
        )

        text = ""
        if hasattr(response, "result"):
            text = str(response.result)
        elif isinstance(response, dict):
            text = response.get("response", response.get("result", ""))

        if not text or not text.strip():
            text = "[No text detected in image. Please type your response instead.]"

        print(f"[OCR] Cloudflare success: extracted {len(text)} characters")
        return {
            "text": text,
            "input_type": "image",
            "success": True,
        }
    except Exception as exc:
        print(f"[OCR] Cloudflare error: {exc}")
        return {
            "text": "[Error: OCR service failed. Please type your response instead.]",
            "input_type": "image",
            "success": False,
            "error": str(exc),
        }


def process_image_from_path(image_path: str) -> Dict[str, Any]:
    """Process an image file through OCR."""
    try:
        image_data_uri = encode_image_to_data_uri(image_path)
        cf_result = _call_cloudflare_ocr(image_data_uri)

        if (
            not cf_result.get("success", False)
            or "No text detected" in cf_result.get("text", "")
            or "[Error" in cf_result.get("text", "")
        ):
            print("[OCR] Cloudflare returned error/no text; falling back to Gemma OCR.")
            return _call_gemma_ocr_from_path(image_path)

        return cf_result
    except APITrackerError:
        raise
    except Exception as exc:
        print(f"[OCR] file-path OCR error: {exc}")
        return {
            "text": "[Error: Could not read image. Please type your response instead.]",
            "input_type": "image",
            "success": False,
            "error": str(exc),
        }


def process_image_from_base64(base64_string: str) -> Dict[str, Any]:
    """Process a base64 encoded image through OCR."""
    tmp_path: Optional[str] = None
    try:
        if base64_string.startswith("data:image/"):
            cf_result = _call_cloudflare_ocr(base64_string)
            if (
                not cf_result.get("success", False)
                or "No text detected" in cf_result.get("text", "")
                or "[Error" in cf_result.get("text", "")
            ):
                print("[OCR] Cloudflare failed for base64 image; falling back to Gemma OCR.")
                base64_data = base64_string.split(",", 1)[1] if "," in base64_string else base64_string
                image_bytes = base64.b64decode(base64_data)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_path = tmp_file.name
                return _call_gemma_ocr_from_path(tmp_path)
            return cf_result

        if "," in base64_string:
            base64_data = base64_string.split(",", 1)[1]
        else:
            base64_data = base64_string

        image_bytes = base64.b64decode(base64_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name

        return process_image_from_path(tmp_path)
    except APITrackerError:
        raise
    except Exception as exc:
        print(f"[OCR] base64 OCR error: {exc}")
        return {
            "text": "[Error: Could not decode image. Please type your response instead.]",
            "input_type": "image",
            "success": False,
            "error": str(exc),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass