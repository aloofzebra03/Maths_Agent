"""
Shared utilities for the Math Tutoring Agent.

Contains helper functions for LLM invocation, JSON extraction, 
and problem loading from JSON files.
"""

import os
import json
import re
from typing import List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import dotenv
dotenv.load_dotenv(dotenv_path=".env", override=True)

import random


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

def get_available_api_keys():
    """Get all available Google API keys from environment."""
    api_keys = []
    for i in range(1, 8):  # Check for GOOGLE_API_KEY1 through GOOGLE_API_KEY7
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            api_keys.append(key)
    
    if not api_keys:
        raise RuntimeError("No Google API keys found. Please set GOOGLE_API_KEY1, GOOGLE_API_KEY2, etc. in .env file")
    
    return api_keys



def get_llm(api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    """
    Get configured LLM instance with specified API key and model.
    
    Args:
        api_key: Google API key. If None, will use environment variable.
        model: Model name to use. Defaults to gemini-2.0-flash-exp.
    
    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    if api_key is None:
        available_keys = get_available_api_keys()
        api_key = random.choice(available_keys)
        print(f"🔑 Selected random API key (ending with ...{api_key[-6:]})")
        if not api_key:
            raise ValueError("No API key provided and GOOGLE_API_KEY not found in environment")
    
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=0.5,
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
    if TRACKER_AVAILABLE:
        # Use tracker to get optimal API key and model
        selected_api_key, selected_model = get_next_available_api_model_pair()
        print(f"🔑 Using tracked API key (ending with ...{selected_api_key[-6:]}) for model: {selected_model}")
        
        # Track BEFORE invocation for accurate rate limiting
        track_model_call(selected_api_key, selected_model)
        
        llm = get_llm(api_key=selected_api_key, model=selected_model)
    else:
        # Fallback to environment variable
        selected_model = "gemini-2.5-flash"
        print(f"🔑 Using default API key from environment for model: {selected_model}")
        llm = get_llm(model=selected_model)
    
    try:
        print(f"▶️ Invoking LLM for operation: {operation_name} with messages: {messages}")
        response = llm.invoke(messages)
        print(f"✅ {operation_name} - Success with model: {selected_model}")
        return response
    except Exception as e:
        print(f"❌ {operation_name} - Failed with model: {selected_model}. Error: {str(e)}")
        raise


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
        Dictionary containing problem data with keys:
            - problem_id: str
            - topic: str
            - difficulty: str
            - question: str
            - final_answer: str
            - canonical_solution: dict with 'steps' list
            - required_concepts: list of str
            - (other optional fields)
    
    Raises:
        FileNotFoundError: If problems_dir doesn't exist
        ValueError: If problem_id not found in any JSON file
    """
    # Get absolute path to problems_json directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems_path = os.path.join(current_dir, problems_dir)
    
    if not os.path.exists(problems_path):
        raise FileNotFoundError(f"Problems directory not found: {problems_path}")
    
    # Search all JSON files for the problem_id
    for filename in os.listdir(problems_path):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(problems_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Handle multiple JSON objects in one file (separated by newlines)
                # Split by '}\\n{' pattern and parse each
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
                
                # Check each parsed object for matching problem_id
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
    """
    Extract step descriptions from canonical solution steps.
    
    Args:
        steps: List of step dictionaries with 'description' field
    
    Returns:
        List of step description strings
    """
    return [step['description'] for step in steps]


def format_required_concepts(concepts: List[str]) -> str:
    """
    Format required concepts list as a readable string.
    
    Args:
        concepts: List of concept identifiers
    
    Returns:
        Formatted string listing concepts
    """
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
    
    This function constructs a properly formatted message list that includes:
    1. System message with core tutor persona (high semantic weight)
    2. Recent conversation history from state["messages"] (preserves roles)
    3. New user instruction/prompt
    4. Optional format instructions
    
    Args:
        state: Agent state containing messages
        system_prompt: Core system instruction (tutor persona, node task, etc.)
        user_prompt: New instruction/query to add
        format_instructions: Optional Pydantic format instructions
        max_history_messages: Maximum number of recent messages to include (default 20)
    
    Returns:
        List of messages ready for LLM invocation
    """
    messages = []
    
    # Add system message with core instruction
    
    # Add recent conversation history from state
    conversation_history = state.get("messages", [])

    if(remove_problem_messages):
        messages.extend(conversation_history[3:])
    else:
        messages.extend(conversation_history)

    messages.append(SystemMessage(content=system_prompt))

    # Add new user instruction
    messages.append(HumanMessage(content=user_prompt))
    
    # Add format instructions if provided
    if format_instructions:
        messages.append(HumanMessage(content=f"\n\n{format_instructions}"))
    
    print(f"📊 Built message list: 1 system + {len(conversation_history)} history + 1 instruction + {1 if format_instructions else 0} format")
    
    return messages
