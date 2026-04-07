"""
Shared utilities for the Math Tutoring Agent.

Contains helper functions for LLM invocation, JSON extraction,
and problem loading from JSON files.
"""

import os
import json
import re
import requests
import uuid
from typing import List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

def extract_json_block(text: str) -> str:
    """
    Extract JSON from text, handling various formats including markdown code blocks.

    Tries three strategies:
    1. Fenced code block with optional 'json' language tag
    2. First balanced JSON object (matching braces)
    3. Return original text (let parser raise error)

    Args:
        text: Raw text that may contain JSON

    Returns:
        Extracted JSON string or original text if no JSON found
    """
    s = text.strip()

    # Strategy 1: Fenced code block (```json {...} ``` or ``` {...} ```)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        result = m.group(1).strip()
        print(f"🎯 JSON extracted from fenced code block")
        return result

    # Strategy 2: Find first balanced JSON object
    start = s.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(s[start:], start=start):
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
                        result = s[start:i+1].strip()
                        print(f"🎯 JSON extracted using balanced braces")
                        return result

    # Strategy 3: No JSON found, return original
    print("⚠️ No JSON found in text, returning original")
    return s


# ============================================================================
# LLM Utilities
# ============================================================================

def get_llm(api_key: Optional[str] = None, model: str = "gemma-3-27b-it") -> ChatGoogleGenerativeAI:
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
        temperature=0.5,
    )


def _normalize_messages_for_model(messages: List, model: str) -> List:
    """
    Normalize a message list so it is compatible with the selected model.

    Gemma models (gemma-3-*) do NOT support SystemMessage — the API returns:
        "400 Developer instruction is not enabled for models/gemma-3-*"

    For these models we convert every SystemMessage by prepending its content
    to the first HumanMessage with a clear header so the LLM still reads the
    instruction as authoritative context.

    Gemini models support SystemMessage natively — no conversion needed.

    Args:
        messages: Message list built by build_messages_with_history()
        model: The model name selected by the tracker

    Returns:
        Normalized message list safe for the given model
    """
    if "gemma" not in model.lower():
        return messages  # Gemini and others — no change needed

    # Collect all SystemMessage content, then rebuild without SystemMessages
    system_parts = []
    other_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(msg.content)
        else:
            other_messages.append(msg)

    if not system_parts:
        return other_messages  # Nothing to convert

    # Build a combined system prefix
    system_prefix = "\n\n---\n\n".join(system_parts)
    system_header = f"[Instructions for you — follow these throughout the conversation]\n{system_prefix}\n\n---\n"

    # Prepend to first HumanMessage, or insert a new HumanMessage at position 0
    if other_messages and isinstance(other_messages[0], HumanMessage):
        other_messages[0] = HumanMessage(
            content=f"{system_header}\n{other_messages[0].content}"
        )
    else:
        other_messages.insert(0, HumanMessage(content=system_header))

    print(f"⚙️ Converted {len(system_parts)} SystemMessage(s) for Gemma compatibility")
    return other_messages


def invoke_llm_with_fallback(messages: List, operation_name: str = "LLM call"):
    """
    Invoke LLM using tracker-selected API key and model pair (if available).

    If tracker is available, automatically selects optimal API key and model
    based on rate limits. Otherwise, uses default API key from environment.

    Before invocation, messages are normalized for the selected model:
    - Gemini models: SystemMessage supported natively (no change)
    - Gemma models: SystemMessage converted to HumanMessage prefix

    Strategy:
    1. Get optimal API key and model from tracker (or use default)
    2. Track the call BEFORE invocation (for rate limiting)
    3. Normalize messages for the selected model
    4. Invoke LLM with selected pair
    5. Let errors bubble up for proper error handling

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
    if TRACKER_AVAILABLE:
        # Use tracker to get optimal API key and model
        selected_api_key, selected_model = get_next_available_api_model_pair()
        print(f"🔑 Using tracked API key (ending with ...{selected_api_key[-6:]}) for model: {selected_model}")

        # Track BEFORE invocation for accurate rate limiting
        track_model_call(selected_api_key, selected_model)

        llm = get_llm(api_key=selected_api_key, model=selected_model)
    else:
        # Fallback to environment variable
        selected_model = "gemma-3-27b-it"
        print(f"🔑 Using default API key from environment for model: {selected_model}")
        llm = get_llm(model=selected_model)

    # Normalize messages for model compatibility (Gemma doesn't support SystemMessage)
    normalized_messages = _normalize_messages_for_model(messages, selected_model)
    print(f"Normalized messages: {normalized_messages}")
    try:
        print(f"▶️ Invoking LLM for operation: {operation_name}")
        response = llm.invoke(normalized_messages)
        print(f"✅ {operation_name} - Success with model: {selected_model}")
        return response
    except Exception as e:
        print(f"❌ {operation_name} - Failed with model: {selected_model}. Error: {str(e)}")
        raise


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
        translated = response.content.strip()
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
            return translate_to_kannada_azure(content)
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
# Conversation History Utilities
# ============================================================================

def build_messages_with_history(
    state: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    format_instructions: Optional[str] = None,
    remove_problem_messages: bool = False,
) -> List:
    """
    Build message list with conversation history for LLM invocation.

    Final message order:
        [SystemMessage(system_prompt)] + [filtered history] + [HumanMessage(user_prompt + format_instructions?)]

    The SystemMessage is ALWAYS first (authoritative context frame): for Gemini
    it is sent natively as a privileged system turn (visible in LangSmith as
    "system"); for Gemma it is converted to a HumanMessage prefix by
    _normalize_messages_for_model before the actual LLM call.

    Stale SystemMessages are ALWAYS stripped from history — every call to this
    function supplies a fresh, node-specific SystemMessage, so any SystemMessage
    left over in state["messages"] from a prior node would create conflicting
    instructions. Filtering them unconditionally keeps the conversation clean.

    user_prompt and format_instructions are merged into ONE final HumanMessage
    because Gemini enforces strict alternating human→model turns; two
    consecutive HumanMessages (with no AI turn between them) are rejected by
    the API. Format instructions are placed LAST inside the turn for maximum
    recency weight.

    Args:
        state: Agent state containing messages
        system_prompt: Core system instruction (tutor persona, node task, etc.)
        user_prompt: Task instruction/query for this turn
        format_instructions: Pydantic format instructions (appended at end of single HumanMessage)
        remove_problem_messages: Kept for backward-compatibility; SystemMessages
            are always stripped from history regardless of this flag.

    Returns:
        List of messages ready for LLM invocation
    """
    # Append language instruction to system prompt
    if state.get("is_kannada", False):
        system_prompt += "\n\nIMPORTANT: You must respond ONLY in Kannada language. All your responses must be in Kannada script, not English."
    else:
        system_prompt += "\n\nIMPORTANT: You must respond ONLY in English. All your responses must be in English, not Kannada or any other language."

    # 1. System message — always first.
    #    For Gemini: sent natively as a system turn (visible in LangSmith as "system").
    #    For Gemma: converted by _normalize_messages_for_model before invocation.
    messages: List = [SystemMessage(content=system_prompt)]

    # 2. Conversation history — always strip stale SystemMessages.
    #    Each call supplies its own fresh system prompt above; retaining old ones
    #    would create conflicting instructions and confuse the model.
    conversation_history = state.get("messages", [])
    filtered_history = [m for m in conversation_history if not isinstance(m, SystemMessage)]
    messages.extend(filtered_history)

    # 3. Single final HumanMessage: task prompt + format instructions (if any).
    #    Merging avoids consecutive HumanMessages which Gemini's API rejects
    #    (it enforces strict alternating human→model turns).
    #    Format instructions go LAST for maximum recency weight.
    final_user_content = user_prompt
    if format_instructions:
        final_user_content = f"{user_prompt}\n\n{format_instructions}"
    messages.append(HumanMessage(content=final_user_content))

    n_history = len(filtered_history)
    print(
        f"📊 Built message list: 1 system + {n_history} history + 1 user turn"
        + (" (includes format spec)" if format_instructions else "")
    )

    return messages
