"""
Shared utilities for the Math Tutoring Agent.

Contains helper functions for LLM invocation, JSON extraction,
and problem loading from JSON files.

Conversation-history strategy (reference pattern):
  Conversation history is serialised to labelled plain text
  (Student: / Agent:), then concatenated with the system prompt,
  task prompt, and format instructions into ONE string that is
  sent as a single HumanMessage.  This is the same technique used
  in educational_agent_optimized_langsmith / shared_utils_reference.py.
"""

import os
import json
import re
import requests
import uuid
from typing import List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import dotenv

dotenv.load_dotenv(dotenv_path=".env", override=True)

# Type alias for AgentState
AgentState = Dict[str, Any]


# Import tracker utilities for API key management
try:
    from api_tracker_utils.tracker import track_model_call, get_next_available_api_model_pair
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    print("⚠️ API tracker not available. Using fallback API key management.")


# ============================================================================
# JSON Extraction Utility
# ============================================================================

def _extract_json_from_str(s: str):
    """
    Try to extract a JSON object from a single string.
    Returns the JSON string on success, or None if not found.

    Scans every '{' in the string as a potential JSON start, validates each
    candidate with json.loads, and returns the first that parses cleanly.
    This avoids false positives like bare '{word}' patterns in reasoning text.
    """
    s = s.strip()

    # Strategy 1: Fenced code block (```json {...} ``` or ``` {...} ```)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Strategy 2: Try every '{' as a potential JSON start, validate each candidate
    i = 0
    while i < len(s):
        if s[i] != "{":
            i += 1
            continue
        # Attempt to find the matching '}' for this '{'
        start = i
        depth = 0
        in_str = False
        esc = False
        j = start
        while j < len(s):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:j + 1].strip()
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break  # This start '{' doesn't yield valid JSON; try next
            j += 1
        i += 1  # Advance past this '{' and try the next one

    return None


def extract_json_block(text) -> str:
    """
    Extract JSON from text, handling various formats including markdown code blocks.

    Tries the following strategies in order:
    1. If input is a list (e.g. gemma-4-31b-it thinking + answer parts), scan
       each part individually from last to first — this avoids picking up
       bare ``{...}`` patterns from reasoning/thinking blocks.
    2. Fenced code block with optional 'json' language tag
    3. First balanced JSON object that is valid JSON
    4. Return original text (let parser raise error)

    Args:
        text: Raw text (str) or list of content parts (e.g. from models that
              return reasoning + answer as separate blocks) that may contain JSON.

    Returns:
        Extracted JSON string or original text if no JSON found
    """
    if isinstance(text, list):
        # Normalise each part to a string
        parts = [
            part if isinstance(part, str) else (part.get("text", "") if isinstance(part, dict) else str(part))
            for part in text
        ]
        # Scan from last part to first — the answer is typically the final element
        for part in reversed(parts):
            result = _extract_json_from_str(part)
            if result is not None:
                print(f"🎯 JSON extracted from list part (reversed scan)")
                return result
        # Fallback: join all parts and try once more
        text = "\n".join(parts)

    result = _extract_json_from_str(text)
    if result is not None:
        print(f"🎯 JSON extracted from text")
        return result

    # No JSON found — return original so parser can raise a meaningful error
    print("⚠️ No JSON found in text, returning original")
    return text if isinstance(text, str) else str(text)


# ============================================================================
# LLM Utilities
# ============================================================================

def get_llm(
    api_key: Optional[str] = None,
    model: str = "gemma-3-27b-it",
    temperature: float = 0.5,
) -> ChatGoogleGenerativeAI:
    """
    Get configured LLM instance with specified API key and model.

    Args:
        api_key: Google API key. If None, will use environment variable.
        model: Model name to use. Defaults to gemma-3-27b-it.

    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY_1")
        if not api_key:
            raise ValueError("No API key provided and GOOGLE_API_KEY not found in environment")

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def invoke_llm_with_fallback(messages: List, operation_name: str = "LLM call"):
    """
    Invoke LLM using tracker-selected API key and model pair (if available).

    If tracker is available, automatically selects optimal API key and model
    based on rate limits. Otherwise, uses default API key from environment.

    Strategy:
    1. Get optimal API key and model from tracker (or use default)
    2. Track the call BEFORE invocation (for rate limiting)
    3. Invoke LLM with selected pair
    4. Let errors bubble up for proper error handling

    Args:
        messages: List of messages to send to the LLM
        operation_name: Name of the operation for logging purposes

    Returns:
        LLM response object

    Raises:
        MinuteLimitExhaustedError: When all API-model pairs hit per-minute limits (if tracker available)
        DayLimitExhaustedError: When all API-model pairs hit daily limits (if tracker available)
        Exception: Any other LLM invocation errors
    """
    temperature = 0.5

    if TRACKER_AVAILABLE:
        selected_api_key, selected_model = get_next_available_api_model_pair()
        print(f"🔑 Using tracked API key (ending with ...{selected_api_key[-6:]}) for model: {selected_model}")
        track_model_call(selected_api_key, selected_model)
        llm = get_llm(api_key=selected_api_key, model=selected_model, temperature=temperature)
    else:
        selected_model = "gemma-3-27b-it"
        print(f"🔑 Using default API key from environment for model: {selected_model}")
        llm = get_llm(model=selected_model, temperature=temperature)

    print(f"📨 Sending {len(messages)} message(s) to model: {selected_model} (temperature={temperature})")
    try:
        print(f"▶️ Invoking LLM for operation: {operation_name}")
        response = llm.invoke(messages)
        print(f"✅ {operation_name} - Success with model: {selected_model}")
        return response
    except Exception as e:
        print(f"❌ {operation_name} - Failed with model: {selected_model}. Error: {str(e)}")
        raise


def translate_to_kannada_google(text: str,
                               source: str = "en",
                               target: str = "kn") -> str:
    """
    Translate text to Kannada using deep_translator GoogleTranslator.

    Args:
        text: Text to translate
        source: Source language code (default: "en")
        target: Target language code (default: "kn")

    Returns:
        Translated text in Kannada.

    Notes:
        Used by translate_if_kannada as the primary Kannada translation backend.
        translate_to_kannada_azure is retained as an alternative if needed.
    """
    try:
        # Lazy import keeps this module import-safe if deep_translator is absent.
        from deep_translator import GoogleTranslator

        translated_text = GoogleTranslator(source=source, target=target).translate(text)
        print(f"✅ GoogleTranslator translation success: {translated_text[:50]}...")
        return translated_text
    except ImportError as e:
        error_msg = (
            "⚠️ deep_translator is not installed. Install with `pip install deep-translator`."
        )
        print(error_msg)
        raise ImportError(error_msg) from e
    except Exception as e:
        print(f"⚠️ GoogleTranslator translation error: {str(e)}")
        raise Exception(f"GoogleTranslator translation error: {str(e)}")


def translate_to_kannada_azure(text: str,
                               api_key: Optional[str] = None,
                               endpoint: str = "https://api.cognitive.microsofttranslator.com/",
                               region: str = "eastasia") -> str:
    """
    Translate text from English to Kannada using Azure Translator.

    Args:
        text: Text to translate (English or mixed English-Kannada)
        api_key: Azure Translator API key. If None, reads from AZURE_TRANSLATOR_KEY env var
        endpoint: Azure Translator endpoint URL
        region: Azure region for the translator service

    Returns:
        Translated text in Kannada, or original text if translation fails
    """
    if api_key is None:
        api_key = os.getenv("AZURE_TRANSLATOR_KEY")

    if not api_key:
        error_msg = "⚠️ Azure Translator API key not found in environment. Set AZURE_TRANSLATOR_KEY in .env file."
        print(error_msg)
        raise ValueError(error_msg)

    try:
        path = '/translate'
        constructed_url = endpoint + path

        params = {
            'api-version': '3.0',
            'from': 'en',
            'to': 'kn'
        }

        headers = {
            'Ocp-Apim-Subscription-Key': api_key,
            'Ocp-Apim-Subscription-Region': region,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        body = [{'text': text}]

        request = requests.post(constructed_url, params=params, headers=headers, json=body, timeout=10)
        response = request.json()

        if request.status_code == 200 and response:
            translated_text = response[0]['translations'][0]['text']
            detected_lang = response[0].get('detectedLanguage', {}).get('language', 'unknown')
            print(f"✅ Translated to Kannada (detected: {detected_lang}): {translated_text[:50]}...")
            return translated_text
        else:
            print(f"⚠️ Azure translation failed with status {request.status_code}: {response}")
            raise Exception(f"Azure translation failed with status {request.status_code}: {response}")

    except Exception as e:
        print(f"⚠️ Azure translation error: {str(e)}. Returning original text.")
        raise Exception(f"Azure translation error: {str(e)}. Returning original text.")


def translate_to_english_gemini(text: str) -> str:
    """
    Translate text from Kannada (or any language) to English using the Gemini API.

    Args:
        text: Text to translate to English.

    Returns:
        Translated text in English, or original text if translation fails.
    """
    msgs = [
        SystemMessage(content=(
            "You are a professional translator. "
            "Translate the following text to English. "
            "Output ONLY the translated text — no explanation, commentary, or extra formatting."
        )),
        HumanMessage(content=text),
    ]

    try:
        response = invoke_llm_with_fallback(msgs, operation_name="Kannada-to-English translation")
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                p if isinstance(p, str) else (p.get("text", "") if isinstance(p, dict) else str(p))
                for p in content
            )
        translated = content.strip()
        print(f"✅ Translated to English: {translated[:80]}...")
        return translated
    except Exception as e:
        print(f"⚠️ Gemini translation error: {str(e)}. Returning original text.")
        return text


def translate_if_kannada(state: AgentState, content: str) -> str:
    """
    Translate content to Kannada if is_kannada flag is set.
    This is the single point of translation - use before setting agent_output.

    Args:
        state: AgentState to check for is_kannada flag
        content: Text to potentially translate

    Returns:
        Translated text if is_kannada=True, otherwise original content
    """
    if state.get("is_kannada", False):
        if re.search(r"[a-zA-Z]", content):
            print("🌐 Translating English response to Kannada...")
            return translate_to_kannada_google(content)
        print("✅ Content is pure Kannada, no translation needed.")
    return content


# ============================================================================
# Problem Loading Utilities
# ============================================================================

def load_problem_from_json(problem_id: str, problems_dir: str = "problems_json") -> Dict[str, Any]:
    """
    Load a problem definition from JSON files.

    Searches all JSON files in the problems_json directory for the specified problem_id.

    Args:
        problem_id: Unique identifier for the problem (e.g., "add_frac_same_den_01")
        problems_dir: Directory containing problem JSON files

    Returns:
        Dictionary containing problem data

    Raises:
        FileNotFoundError: If problems_dir doesn't exist
        ValueError: If problem_id not found in any JSON file
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems_path = os.path.join(current_dir, problems_dir)

    if not os.path.exists(problems_path):
        raise FileNotFoundError(f"Problems directory not found: {problems_path}")

    for filename in os.listdir(problems_path):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(problems_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

                json_objects = []
                current_obj = ""
                brace_count = 0

                for char in content:
                    current_obj += char
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and current_obj.strip():
                            try:
                                obj = json.loads(current_obj.strip())
                                json_objects.append(obj)
                                current_obj = ""
                            except json.JSONDecodeError:
                                pass

                for problem_data in json_objects:
                    if problem_data.get('problem_id') == problem_id:
                        print(f"✅ Loaded problem '{problem_id}' from {filename}")
                        print(f"📚 Problem data: {problem_data}")
                        return problem_data

        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            continue

    raise ValueError(f"Problem ID '{problem_id}' not found in {problems_path}")


def get_step_descriptions(steps: List[Dict[str, Any]]) -> List[str]:
    """Extract step descriptions from canonical solution steps."""
    return [step['description'] for step in steps]


def format_required_concepts(concepts: List[str]) -> str:
    """Format required concepts list as a readable string."""
    if not concepts:
        return "None specified"
    return ", ".join(concepts)


# ============================================================================
# Conversation History Utilities  (reference / text-serialisation pattern)
# ============================================================================

def build_conversation_history(state: AgentState) -> str:
    """
    Serialise state["messages"] to a plain-text labelled transcript.

    Each message type gets a clear speaker label:
        Student: <human turn>
        Agent:   <ai turn>
        System:  <system turn>  (rarely present; kept for completeness)

    The "__start__" sentinel used by some graph initialisers is skipped.

    Args:
        state: Agent state containing messages list

    Returns:
        Newline-separated labelled transcript, or "" if history is empty
    """
    conversation = state.get("messages", [])
    history_text = ""

    for msg in conversation:
        if isinstance(msg, HumanMessage):
            if msg.content == "__start__":
                continue
            history_text += f"Student: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Agent: {msg.content}\n"
        elif isinstance(msg, SystemMessage):
            history_text += f"System: {msg.content}\n"

    return history_text.strip()


def build_messages_with_history(
    state: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    format_instructions: Optional[str] = None,
    remove_problem_messages: bool = False,
    include_last_message: bool = False,
) -> List:
    """
    Build a single-message list for LLM invocation using the reference pattern.

    Everything — system prompt, conversation history, last student message
    (optional), task prompt, and format instructions — is assembled into ONE
    plain-text string and wrapped in a single HumanMessage.

    Assembled string order (mirrors build_prompt_from_template in reference):
        {system_prompt + language instruction}

        Conversation History:
        Student: ...
        Agent:   ...

        Student's Latest Response: {last_user_msg}   ← only if include_last_message=True

        {user_prompt}

        {format_instructions}    ← appended last, maximum recency weight

    Args:
        state: Agent state containing messages
        system_prompt: Core system instruction (tutor persona, node task, etc.)
        user_prompt: Task context / question for this turn
        format_instructions: Pydantic format instructions (appended at end)
        remove_problem_messages: Kept for backward-compatibility; no effect in
            the text-serialisation pattern.
        include_last_message: If True and state["last_user_msg"] is set, appends
            the student's latest response explicitly — identical to the
            include_last_message flag in the reference's build_prompt_from_template.

    Returns:
        [HumanMessage(combined_prompt_string)]  — single-element list compatible
        with invoke_llm_with_fallback()
    """
    # Append language instruction to system prompt
    if state.get("is_kannada", False):
        system_prompt += "\n\nIMPORTANT: You must respond ONLY in Kannada language. All your responses must be in Kannada script, not English."
    else:
        system_prompt += "\n\nIMPORTANT: You must respond ONLY in English. All your responses must be in English, not Kannada or any other language."

    # Build conversation history as labelled text
    history = build_conversation_history(state)

    # Assemble prompt parts using a PromptTemplate (mirrors reference exactly)
    template_parts = ["{system_prompt}"]
    template_vars = ["system_prompt"]

    # Add history if available
    if history:
        template_parts.append("\n\nConversation History:\n{history}")
        template_vars.append("history")

    # Add last user message if requested (reference pattern)
    if include_last_message and state.get("last_user_msg"):
        template_parts.append("\n\nStudent's Latest Response: {last_user_message}")
        template_vars.append("last_user_message")

    # Add the task prompt
    template_parts.append("\n\n{user_prompt}")
    template_vars.append("user_prompt")

    # Add output contract and format instructions at the end for maximum recency weight.
    # Keep format_instructions as the absolute final content seen by the LLM.
    if format_instructions:
        template_parts.append(
            "\n\nIMPORTANT OUTPUT CONTRACT:\n"
            "- Return ONLY a valid JSON object.\n"
            "- Do NOT include markdown code fences.\n"
            "- Do NOT include any extra prose before or after JSON."
        )
        template_parts.append("\n\n{format_instructions}")
        template_vars.append("format_instructions")

    template_string = "".join(template_parts)
    prompt_template = PromptTemplate(
        input_variables=template_vars,
        template=template_string,
    )

    template_values: Dict[str, str] = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    if history:
        template_values["history"] = history
    if include_last_message and state.get("last_user_msg"):
        template_values["last_user_message"] = state["last_user_msg"]
    if format_instructions:
        template_values["format_instructions"] = format_instructions

    final_prompt = prompt_template.format(**template_values)

    n_history = len(state.get("messages", []))
    print(
        f"📊 Built prompt: system + {n_history} history msg(s) + task prompt"
        + (" + last_user_msg" if include_last_message and state.get("last_user_msg") else "")
        + (" + format spec" if format_instructions else "")
    )
    print(f"📝 Prompt length: {len(final_prompt)} characters")

    return [HumanMessage(content=final_prompt)]


# OCR helpers live in utils.ocr_utilities and are re-exported here for
# backward compatibility with existing imports.
from utils.ocr_utilities import (  # noqa: E402
    encode_image_to_data_uri,
    process_image_from_path,
    process_image_from_base64,
)