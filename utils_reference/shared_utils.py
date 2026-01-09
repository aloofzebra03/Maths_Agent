# shared_utils.py
"""
Shared utilities for educational agents.
Contains common helper functions used by both traditional nodes and simulation nodes.
"""

import os
import json
import re
import random
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# from educational_agent_v1.config_rag import concept_pkg
# from educational_agent_v1.Creating_Section_Text.retriever import retrieve_docs
# from educational_agent_v1.Filtering_GT.filter_utils import filter_relevant_section
# from educational_agent_v1.Creating_Section_Text.schema import NextSectionChoice

from api_tracker_utils.tracker import track_model_call,get_next_available_api_model_pair
from api_tracker_utils.config import AVAILABLE_MODELS, DEFAULT_MODEL

dotenv.load_dotenv(dotenv_path=".env", override=True)

# Type alias for AgentState - flexible to work with different state structures
AgentState = Dict[str, Any]

# ─── Autosuggestion Pool Constants (Single Source of Truth) ──────────────────

# Positive affirmations - student understands/agrees
POSITIVE_POOL = [
    "I understand, continue",
    "Yes, got it",
    "That makes sense",
    "Let's proceed further",
    "I'm following along",
    None
]

# Negative/uncertainty - student confused/needs help
NEGATIVE_POOL = [
    "I'm not sure",
    "I don't know",
    "I'm confused",
    "Not very clear",
    "Can you explain differently?"
]

# Special handling - triggers handler logic
SPECIAL_HANDLING_POOL = [
    "Can you give me a hint?",
    "Can you explain that simpler?",
    "Give me an example"
]

# ─────────────────────────────────────────────────────────────────────

def extract_json_block(text: str) -> str:
    """Extract JSON from text, handling various formats including markdown code blocks."""
    s = text.strip()

    # 🔍 JSON EXTRACTION INPUT 🔍
    print("=" * 60)
    print("🔧 JSON EXTRACTION - INPUT TEXT")
    print("=" * 60)
    print(f"📄 INPUT_LENGTH: {len(s)} characters")
    print(f"📄 INPUT_PREVIEW: {s[:200]}...")
    print("=" * 60)

    # 1) Try to find a fenced code block containing JSON (language tag optional)
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        result = m.group(1).strip()
        print("🎯 JSON EXTRACTED - METHOD: Fenced code block")
        print(f"📦 EXTRACTED_JSON: {result}")
        print("=" * 60)
        return result

    # 2) Try to find the first balanced JSON object in the text
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
                        print("🎯 JSON EXTRACTED - METHOD: Balanced braces")
                        print(f"📦 EXTRACTED_JSON: {result}")
                        print("=" * 60)
                        return result

    # 3) Nothing found — return original (let parser raise)
    print("⚠️ JSON EXTRACTION - METHOD: No JSON found, returning original")
    print(f"📦 RETURNED_TEXT: {s}")
    print("=" * 60)
    return s


def get_llm(api_key: Optional[str] = None, model: str = "gemma-3-27b-it"):
    """Get configured LLM instance with specified or chosen API key and model.
    
    Args:
        api_key: Google API key. If None, randomly selects from available keys.
        model: Model name to use. Defaults to gemma-3-27b-it.
    
    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=0.5,
    )
    # llm = ChatGroq(
    #     model="llama-3.1-8b-instant",
    #     temperature=0.5,
    #     max_tokens=None,
    # )
    # return llm


def invoke_llm_with_fallback(messages: List, operation_name: str = "LLM call"):
    """
    Invoke LLM using tracker-selected API key and model pair.
    
    The tracker automatically selects the optimal API key and model combination
    based on rate limits and load balancing. No manual model selection is supported.
    
    Strategy:
    1. Get optimal API key and model from tracker (respects rate limits)
    2. Track the call BEFORE invocation (for rate limiting)
    3. Invoke LLM with selected pair
    4. Let errors bubble up (MinuteLimitExhaustedError, DayLimitExhaustedError, etc.)
    
    Args:
        messages: List of messages to send to the LLM
        operation_name: Name of the operation for logging purposes
    
    Returns:
        LLM response object
    
    Raises:
        MinuteLimitExhaustedError: When all API-model pairs hit per-minute limits
        DayLimitExhaustedError: When all API-model pairs hit daily limits
        Exception: Any other LLM invocation errors
    """
    # Get optimal pair from tracker (may raise rate limit errors)
    selected_api_key, selected_model = get_next_available_api_model_pair()
    print(f"🔑 Using tracked API key (ending with ...{selected_api_key[-6:]}) for model: {selected_model}")

    # Track BEFORE invocation for accurate rate limiting
    track_model_call(selected_api_key, selected_model)
    
    # Create LLM instance and invoke
    llm = get_llm(api_key=selected_api_key, model=selected_model)
    
    try:
        response = llm.invoke(messages)
        print(f"✅ {operation_name} - Success with tracked key and model: {selected_model}")
        return response
    except Exception as e:
        print(f"❌ {operation_name} - Failed with tracked key/model: {selected_model}. Error: {str(e)}")
        raise


def add_ai_message_to_conversation(state: AgentState, content: str):
    """Add AI message to conversation after successful processing."""
    state["messages"].append(AIMessage(content=content))
    print(f"📝 Added AI message to conversation: {content[:50]}...")


def add_system_message_to_conversation(state: AgentState, content: str):
    """Add System message to conversation after successful processing."""
    state["messages"].append(SystemMessage(content=content))
    print(f"📝 Added System message to conversation: {content[:50]}...")


def llm_with_history(state: AgentState, final_prompt: str):
    # 🔍 LLM INVOCATION - INPUT 🔍
    print("=" * 70)
    print("🤖 LLM INVOCATION - STARTED")
    print("=" * 70)
    print(f"📝 PROMPT_LENGTH: {len(final_prompt)} characters")
    print(f"📝 PROMPT_PREVIEW: {final_prompt}...")
    print("=" * 70)
    
    # Send the final prompt directly as a human message
    # Note: The final_prompt already contains conversation history via build_prompt_from_template
    request_msgs = [HumanMessage(content=final_prompt)]
    
    # Use the centralized invoke function - tracker will select optimal model
    resp = invoke_llm_with_fallback(request_msgs, operation_name="LLM with history")
    
    # 🔍 LLM INVOCATION - OUTPUT 🔍
    print("🤖 LLM INVOCATION - COMPLETED")
    print(f"📤 RESPONSE_LENGTH: {len(resp.content)} characters")
    print(f"📤 RESPONSE_PREVIEW: {resp.content[:200]}...")
    print(f"📊 RESPONSE_TYPE: {type(resp).__name__}")
    print("=" * 70)
    
    # DO NOT append to messages here - let the calling node handle it after parsing
    return resp


def build_conversation_history(state: AgentState) -> str:
    conversation = state.get("messages", [])
    history_text = ""
    
    for msg in conversation:
        if isinstance(msg, HumanMessage) and msg.content == "__start__":
            continue
        elif isinstance(msg, HumanMessage):
            history_text += f"Student: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Agent: {msg.content}\n"
        elif isinstance(msg, SystemMessage):
            history_text += f"System: {msg.content}\n"
    
    return history_text.strip()


def build_prompt_from_template(system_prompt: str, state: AgentState, 
                             include_last_message: bool = False, 
                             include_instructions: bool = False,
                             parser=None) -> str:
    
    # Add Kannada instruction if needed
    if state.get("is_kannada", False):
        system_prompt += "\n\nIMPORTANT: You must respond ONLY in Kannada language. All your responses must be in Kannada script, not English."
    
    # Build the template string based on what we need
    template_parts = ["{system_prompt}"]
    template_vars = ["system_prompt"]
    
    # Add history if available
    history = build_conversation_history(state)
    if history:
        template_parts.append("\n\nConversation History:\n{history}")
        template_vars.append("history")
    
    # Add last user message if requested
    if include_last_message and state.get("last_user_msg"):
        template_parts.append("\n\nStudent's Latest Response: {last_user_message}")
        template_vars.append("last_user_message")
    
    # Add instructions at the end if requested
    if include_instructions and parser:
        template_parts.append("\n\n{instructions}")
        template_vars.append("instructions")
    
    # Create the template
    template_string = "".join(template_parts)
    prompt_template = PromptTemplate(
        input_variables=template_vars,
        template=template_string
    )
    
    # Prepare the values
    template_values = {"system_prompt": system_prompt}
    
    if history:
        template_values["history"] = history
    
    if include_last_message and state.get("last_user_msg"):
        template_values["last_user_message"] = state["last_user_msg"]
    
    if include_instructions and parser:
        template_values["instructions"] = parser.get_format_instructions()
    
    # Format the prompt
    return prompt_template.format(**template_values)


def build_prompt_from_template_optimized(system_prompt: str, state: AgentState, 
                                       include_last_message: bool = False, 
                                       include_instructions: bool = False,
                                       parser=None, current_node: str = None,
                                       include_autosuggestions: bool = True) -> str:
    
    # Add Kannada instruction if needed
    if state.get("is_kannada", False):
        system_prompt += "\n\nIMPORTANT: You must respond ONLY in Kannada language. All your responses must be in Kannada script, not English."
    
    # Build the template string based on what we need
    template_parts = ["{system_prompt}"]
    template_vars = ["system_prompt"]
    
    # Call history building functions once and reuse the result
    if current_node:
        history = build_node_aware_conversation_history(state, current_node)
    else:
        # Fall back to regular history if no current_node provided
        history = build_conversation_history(state)
    
    # Add history to template if available
    if history:
        template_parts.append("\n\nConversation History:\n{history}")
        template_vars.append("history")
    
    # Add last user message if requested
    if include_last_message and state.get("last_user_msg"):
        print("=====================================================")
        print("Adding last user message to prompt template")
        print("Last user message:", state["last_user_msg"])
        print("=====================================================")
        template_parts.append("\n\nStudent's Latest Response: {last_user_message}")
        template_vars.append("last_user_message")
    
    # Add autosuggestion instructions BEFORE format instructions for pedagogical nodes
    if include_autosuggestions and parser and current_node in ["APK", "CI", "GE", "AR", "TC", "RLC"]:
        # Use imported pool constants for prompt generation
        positive_pool = POSITIVE_POOL
        negative_pool = NEGATIVE_POOL
        special_handling_pool = SPECIAL_HANDLING_POOL
        
        # Get student level and corresponding description
        student_level = state.get("student_level", "medium")
        level_descriptions = {
            "low": "struggling student who needs extra scaffolding, simpler language, and encouragement",
            "medium": "average student progressing normally who needs moderate guidance",
            "advanced": "excelling student ready for deeper challenges and critical thinking"
        }
        level_desc = level_descriptions.get(student_level, level_descriptions["medium"])
        
        template_parts.append(f"""\n\nIMPORTANT - Autosuggestion Generation:

ANALYZE THE CONVERSATION CONTEXT:
- Review the conversation history above carefully
- Consider your current feedback/message that you're about to send
- Determine if your message is a QUESTION or contains "let me think"
- Select autosuggestions that make sense given where the student is in their learning journey
- Make suggestions relevant to what you just explained or asked

🚨 CRITICAL: QUESTION/THINKING DETECTION 🚨
IF your feedback contains ANY of the following:
  • A direct question to the student (e.g., "Can you tell me..?", "What do you think..?", "Why does..?")
  • The phrase "let me think"
  • Asking student to explain, describe, or answer something

THEN you MUST:
  • Set positive_autosuggestion = null/None (not a string, use JSON null)
  • Set dynamic_autosuggestion = null/None (not a string, use JSON null)
  • Only provide negative_autosuggestion and special_handling_autosuggestion

Reason: When asking questions, we don't want suggestions like "I understand" or exploratory prompts that might distract from answering the question.

SELECTION RULES:

1. **positive_autosuggestion** - CONDITIONAL:
   
   IF your message is NOT a question and does NOT contain "let me think":
       
   - Pick the most contextually appropriate positive/affirmative option from the list below:
     → Select EXACTLY ONE from positive pool: {positive_pool}
     ⚠️ WARNING: Only choose from the positive pool above, even if student is confused!
     → Pick the most contextually appropriate positive/affirmative option
     → These represent what a student COULD say if they understand/agree
   
   IF your message IS a question OR contains "let me think":
     → Set to null/Select None (JSON null, not string "null")
   
2. **negative_autosuggestion** - ALWAYS REQUIRED:
   - Pick the most contextually appropriate negative/uncertain option from the list below:
   → Select EXACTLY ONE from negative pool: {negative_pool}
       ⚠️ WARNING: Only choose from the negative pool above, even if student is confused!
   → Pick the most contextually appropriate negative/uncertain option
   → These represent what a student COULD say if they're confused or need help
   
3. **special_handling_autosuggestion** - ALWAYS REQUIRED:
   → Select EXACTLY ONE from special handling pool: {special_handling_pool}
   → This will trigger special pedagogical intervention (hints, examples, simpler explanation)
   → Choose based on what type of help would be most useful given your current message:
     • "Can you give me a hint?" - for nudging without revealing answer
     • "Can you explain that simpler?" - for complex explanations
     • "Give me an example" - for abstract concepts
   
4. **dynamic_autosuggestion** - CONDITIONAL:
   
   IF your message is NOT a question and does NOT contain "let me think":
     → Generate EXACTLY ONE unique exploratory suggestion (12-15 words max):
       • Must be contextually relevant to the CURRENT conversation and your message
       • Should point to a specific unexplored aspect related to what you just explained
       • Should nudge student to think about a related concept/application/implication
       • Must be DIFFERENT from all pool suggestions above
       • Use noun-phrase or question format that evokes curiosity
       
       Adjust depth based on student level ({student_level} - {level_desc}):
       
       • low: Concrete, visible aspects
         - Focus on: where it happens, what is used/made, which part does it
         - Example: "Where exactly in the leaf does this happen?"
         - Avoid: abstraction, complex variations, dependencies
       
       • medium: Cause-effect, constraints
         - Focus on: why needed, what enables/prevents, usefulness, limitations
         - Example: "Why only green parts of a plant can do this"
         - Balance: not too simple, not overly complex
       
       • advanced: Dependencies, variations, implications
         - Focus on: how changes affect outcomes, limiting factors, broader impact
         - Example: "How changes in sunlight intensity affect the rate"
         - Encourage: critical thinking about relationships and constraints
   
   IF your message IS a question OR contains "let me think":
     → Set to null/None (JSON null, not string "null")

CRITICAL CONTEXT AWARENESS:
- First, check if your feedback is a question → if yes, set positive and dynamic to null
- If you just explained something complex → lean towards selecting confused/uncertain options for negative
- If student just answered correctly → this is explanatory feedback, provide all 4 suggestions
- If you're asking a challenging question → positive and dynamic must be null, only negative and special
- If you're providing encouragement → this is feedback, provide all 4 suggestions
- Make suggestions relate to the CURRENT pedagogical moment in the conversation

REMEMBER:
- Questions/"let me think" → 2 suggestions (negative + special), positive and dynamic are null
- Explanations/Feedback → 4 suggestions (positive + negative + special + dynamic)
- Set as None""")

    
    # Add instructions at the end if requested
    if include_instructions and parser:
        template_parts.append("\n\n{instructions}")
        template_vars.append("instructions")
    
    # Create the template
    template_string = "".join(template_parts)
    prompt_template = PromptTemplate(
        input_variables=template_vars,
        template=template_string
    )
    
    # Prepare the values
    template_values = {"system_prompt": system_prompt}
    
    # Add history if available (already computed above)
    if history:
        template_values["history"] = history
    
    if include_last_message and state.get("last_user_msg"):
        template_values["last_user_message"] = state["last_user_msg"]
    
    if include_instructions and parser:
        template_values["instructions"] = parser.get_format_instructions()
    
    # Format the prompt
    return prompt_template.format(**template_values)


def get_ground_truth(concept: str, section_name: str) -> str:
    # """Retrieve ground truth content for a given concept and section."""
    # try:
    #     # 🔍 GROUND TRUTH RETRIEVAL - INPUT 🔍
    #     print("=" * 70)
    #     print("📚 GROUND TRUTH RETRIEVAL - STARTED")
    #     print("=" * 70)
    #     print(f"🎯 CONCEPT: {concept}")
    #     print(f"📋 SECTION_NAME: {section_name}")
    #     print("=" * 70)
        
    #     # Build a minimal NextSectionChoice object; other fields are dummy since retriever only uses section_name
    #     params = NextSectionChoice(
    #         section_name=section_name,
    #         difficulty=1,
    #         board_exam_importance=1,
    #         olympiad_importance=1,
    #         avg_study_time_min=1,
    #         interest_evoking=1,
    #         curiosity_evoking=1,
    #         critical_reasoning_needed=1,
    #         inquiry_learning_scope=1,
    #         example_availability=1,
    #     )
    #     docs = retrieve_docs(concept, params)
    #     combined = [f"# Page: {d.metadata['page_label']}\n{d.page_content}" for d in docs]
    #     full_doc = "\n---\n".join(combined)
    #     result = filter_relevant_section(concept, section_name, full_doc)
        
    #     # 🔍 GROUND TRUTH RETRIEVAL - OUTPUT 🔍
    #     print("📚 GROUND TRUTH RETRIEVAL - COMPLETED")
    #     print(f"📄 DOC_COUNT: {len(docs)} documents")
    #     print(f"📏 FULL_DOC_LENGTH: {len(full_doc)} characters")
    #     print(f"📏 FILTERED_LENGTH: {len(result)} characters")
    #     print(f"📄 RESULT_PREVIEW: {result[:300]}...")
    #     print("=" * 70)
        
    #     return result
    # except Exception as e:
    #     print(f"Error retrieving ground truth for {concept} - {section_name}: {e}")
    #     raise
    return ""


# ─────────────────────────────────────────────────────────────────────
# Concept-to-JSON mapping cache
# ─────────────────────────────────────────────────────────────────────

_CONCEPT_TO_FILE_MAP = None  # Cache for concept-to-file mapping

# Concept alias mapping - maps display names to actual JSON concept names
# This handles cases where the user-facing name differs from the JSON key
_CONCEPT_ALIAS_MAP = {
    "pendulum and its time period": "measurement of time",
    # Add more aliases as needed
}

# Hardcoded section key mapping - covers all possible variations across JSON files
_SECTION_KEY_MAPPING = {
    # Concept definition/description
    "concept definition": ["description", "Description", "desc"],
    
    # Explanation with analogies/intuition
    "explanation (with analogies)": [
        "intuition_logical_flow", 
        "Intuition_Logical_Flow", 
        "Intuition / Logical Flow",
        "intuition / logical flow",
        "Intuition/Logical Flow"
    ],
    
    # Detailed information
    "details (facts, sub-concepts)": ["detail", "Detail", "details", "Details"],
    
    # MCQs - multiple choice questions
    "mcqs": [
        "open_ended_mcqs", 
        "Open-Ended_MCQs", 
        "Open-Ended MCQs",
        "open-ended mcqs",
        "Open Ended MCQs",
        "mcqs",
        "MCQs"
    ],
    
    # Real-life applications
    "real-life application": [
        "real_life_applications", 
        "Real-Life_Applications", 
        "Real-Life Applications",
        "real life applications",
        "Real Life Applications",
        "real-life applications"
    ],
    
    # Working/how it works
    "working": ["working", "Working", "how_it_works", "How_It_Works"],
    
    # Critical thinking
    "critical thinking": [
        "critical_thinking", 
        "Critical_Thinking",
        "critical thinking",
        "Critical Thinking"
    ],
    
    # Key topics from textbook
    "key topics": [
        "key_topics_from_the_textbook",
        "Key_Topics_from_the_Textbook",
        "Key Topics from the Textbook",
        "key topics from the textbook",
        "key_topics",
        "Key_Topics",
        "Key Topics from Textbook"
    ],
    
    # Exam-oriented questions
    "exam questions": [
        "exam_oriented_questions",
        "Exam-Oriented_Questions",
        "Exam-Oriented Questions",
        "exam oriented questions",
        "Exam Oriented Questions"
    ],
    
    # Cross-concept critical thinking
    "cross-concept thinking": [
        "cross_concept_critical_thinking",
        "Cross-Concept_Critical_Thinking",
        "Cross-Concept Critical Thinking",
        "cross concept critical thinking"
    ],
    
    # Relation between sub-concepts
    "relations": [
        "relation_between_sub_concepts",
        "Relation_Between_Sub-Concepts",
        "Relation Between Sub-Concepts",
        "relation between sub-concepts"
    ],
    
    # What-if scenarios
    "what-if scenarios": [
        "what_if_scenarios",
        "What-if_Scenarios",
        "What-If Scenarios",
        "what if scenarios"
    ],
}

def _build_concept_to_file_mapping() -> Dict[str, str]:
    """
    Hardcoded mapping of concept names (lowercase) to their JSON file paths.
    This eliminates the need to scan files on every import, providing instant lookups.
    
    Returns:
        Dict mapping concept names to file paths
    """
    global _CONCEPT_TO_FILE_MAP
    
    if _CONCEPT_TO_FILE_MAP is not None:
        return _CONCEPT_TO_FILE_MAP
    
    # Hardcoded mapping - add new concepts here as needed
    mapping = {
        # 8.json concepts
        "conduction": "NCERT/8.json",
        "convection": "NCERT/8.json",
        "radiation": "NCERT/8.json",
        "good conductors of heat": "NCERT/8.json",
        "poor conductors (insulators) of heat": "NCERT/8.json",
        "land breeze": "NCERT/8.json",
        "sea breeze": "NCERT/8.json",
        "water cycle": "NCERT/8.json",
        "infiltration": "NCERT/8.json",
        "groundwater": "NCERT/8.json",
        
        # 11.json concepts
        "photosynthesis": "NCERT/11.json",
        "chlorophyll": "NCERT/11.json",
        "stomata": "NCERT/11.json",
        "xylem": "NCERT/11.json",
        "phloem": "NCERT/11.json",
        "respiration": "NCERT/11.json",
        "carbon dioxide": "NCERT/11.json",
        "oxygen": "NCERT/11.json",
        "water": "NCERT/11.json",
        "sunlight": "NCERT/11.json",
        
        # 12.json concepts
        "light": "NCERT/12.json",
        "shadows": "NCERT/12.json",
        "reflection": "NCERT/12.json",
        "luminous objects": "NCERT/12.json",
        "non-luminous objects": "NCERT/12.json",
        "transparent materials": "NCERT/12.json",
        "translucent materials": "NCERT/12.json",
        "opaque materials": "NCERT/12.json",
        "image formation": "NCERT/12.json",
        "pinhole camera": "NCERT/12.json",
        
        # NCERT Class 7.json concepts
        "measurement of time": "NCERT/NCERT Class 7.json",
        "history of timekeeping devices": "NCERT/NCERT Class 7.json",
        "sundial": "NCERT/NCERT Class 7.json",
        "water clock": "NCERT/NCERT Class 7.json",
        "hourglass": "NCERT/NCERT Class 7.json",
        "candle clock": "NCERT/NCERT Class 7.json",
        "pendulum and its time period": "NCERT/NCERT Class 7.json",
        "speed": "NCERT/NCERT Class 7.json",
        "uniform motion": "NCERT/NCERT Class 7.json",
        "non-uniform motion": "NCERT/NCERT Class 7.json",
        "si unit of time": "NCERT/NCERT Class 7.json",

        # Fractions.txt concepts
        "fraction as equal share": "NCERT/Fractions.txt",
        "fractional units": "NCERT/Fractions.txt",
        "reading fractions": "NCERT/Fractions.txt",
        "numerator": "NCERT/Fractions.txt",
        "denominator": "NCERT/Fractions.txt",
        "mixed fractions": "NCERT/Fractions.txt",
        "number line": "NCERT/Fractions.txt",
        "equivalent fractions": "NCERT/Fractions.txt",
        "lowest terms": "NCERT/Fractions.txt",
        "brahmagupta's method for adding fractions": "NCERT/Fractions.txt",
        "brahmagupta's method for subtracting fractions": "NCERT/Fractions.txt",
        
        "_default": "NCERT/8.json",
    }
    
    _CONCEPT_TO_FILE_MAP = mapping
    print(f"✅ Loaded hardcoded mapping for {len(mapping)-1} concepts")
    return mapping


def get_all_available_concepts() -> List[str]:
    """
    Get list of all available concepts from the mapping.
    
    Returns:
        List of concept names (properly capitalized for display)
    """
    mapping = _build_concept_to_file_mapping()
    
    # Get all concept keys except the default
    concepts = [key for key in mapping.keys() if key != "_default"]
    
    # Sort alphabetically for better UX
    concepts.sort()
    
    return concepts


def _extract_concept_data_from_json(data: dict, concept: str) -> Optional[dict]:
    """
    Extract concept data from JSON with structure: {"concepts": [{...}, ...]}.
    
    Args:
        data: The loaded JSON data
        concept: The concept name to find
    
    Returns:
        The concept data dict or None if not found
    """
    concept_lower = concept.lower().strip()
    
    # Structure: {"concepts": [{...}, ...]}
    if "concepts" in data and isinstance(data["concepts"], list):
        for concept_data in data["concepts"]:
            concept_name = concept_data.get("concept", concept_data.get("Concept", ""))
            if concept_name.lower().strip() == concept_lower:
                return concept_data
    
    return None


_TEXT_FILE_CACHE = {}

def _parse_text_concept_file(file_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parses a text file with ### Concept: structure into a dictionary.
    Returns: {concept_name_lower: {section_header_lower: content}}
    """
    global _TEXT_FILE_CACHE
    if file_path in _TEXT_FILE_CACHE:
        return _TEXT_FILE_CACHE[file_path]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        concepts = {}
        # Split by "### Concept:"
        parts = re.split(r'### Concept:\s*', text, flags=re.IGNORECASE)
        
        for part in parts[1:]: # Skip the first part
            lines = part.split('\n')
            concept_name = lines[0].strip().lower()
            
            # Split by "#### " to get sections
            content_text = '\n'.join(lines[1:])
            section_parts = re.split(r'####\s*', content_text)
            
            sections = {}
            for section_part in section_parts:
                if not section_part.strip():
                    continue
                
                section_lines = section_part.split('\n')
                section_header = section_lines[0].strip().lower()
                section_content = '\n'.join(section_lines[1:]).strip()
                
                sections[section_header] = section_content
                
            concepts[concept_name] = sections
            
        _TEXT_FILE_CACHE[file_path] = concepts
        print(f"✅ Parsed and cached text file: {file_path} ({len(concepts)} concepts)")
        return concepts
    except Exception as e:
        print(f"❌ Error parsing text file {file_path}: {e}")
        return {}


def get_ground_truth_from_json(concept: str, section_name: str) -> str:
    """
    Retrieve ground truth content from JSON or Text file for a given concept and section.
    Uses cached mapping for fast lookup. No formatting - returns raw content for LLM consumption.
    
    Args:
        concept: The concept name to find
        section_name: The section/key within the concept to retrieve
    
    Returns:
        str: The relevant content from the file
    """
    try:        
        # 🔍 GROUND TRUTH RETRIEVAL - INPUT 🔍
        print("=" * 70)
        print("📚 GROUND TRUTH RETRIEVAL - STARTED")
        print("=" * 70)
        print(f"🎯 CONCEPT: {concept}")
        print(f"📋 SECTION_NAME: {section_name}")
        print("=" * 70)
        
        # Get concept-to-file mapping
        mapping = _build_concept_to_file_mapping()
        concept_key = concept.lower().strip()
        
        print(f"✅ Loaded hardcoded mapping for {len(mapping)} concepts")
        
        # Find the file for this concept
        file_path = mapping.get(concept_key)
        
        if not file_path:
            # Return empty string if concept not found in mapping
            print(f"⚠️ Concept '{concept}' not in mapping, returning empty string")
            print("=" * 70)
            return ""
        
        print(f"📂 Found concept in: {file_path}")
        
        # Handle Text files
        if file_path.endswith('.txt'):
            parsed_data = _parse_text_concept_file(file_path)
            
            if concept_key not in parsed_data:
                result = f"Concept '{concept}' not found in text file"
                print(f"❌ {result}")
                print("=" * 70)
                return ""
            
            concept_sections = parsed_data[concept_key]
            
            # Use the hardcoded section key mapping
            section_key_mapping = _SECTION_KEY_MAPPING
            
            # Get mapped keys (try multiple possible keys)
            possible_keys = section_key_mapping.get(section_name.lower(), [section_name])
            if not isinstance(possible_keys, list):
                possible_keys = [possible_keys]
            
            # Try each possible key until we find content
            content = None
            used_key = None
            
            # Also try exact match with section_name
            keys_to_try = possible_keys + [section_name]
            
            for key in keys_to_try:
                key_lower = key.lower()
                if key_lower in concept_sections:
                    content = concept_sections[key_lower]
                    used_key = key
                    break
            
            if content is None:
                result = f"Section '{section_name}' not found for concept '{concept}' in text file"
            else:
                result = content
                
        # Handle JSON files
        else:
            # Load the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract concept data using the concept name
            concept_data = _extract_concept_data_from_json(data, concept)
            
            if not concept_data:
                result = f"Concept '{concept}' not found in JSON data"
                print(f"❌ {result}")
                print("=" * 70)
                return ""
            
            # Use the hardcoded section key mapping
            section_key_mapping = _SECTION_KEY_MAPPING
            
            # Get mapped keys (try multiple possible keys)
            possible_keys = section_key_mapping.get(section_name.lower(), [section_name])
            if not isinstance(possible_keys, list):
                possible_keys = [possible_keys]
            
            # Try each possible key until we find content
            content = None
            used_key = None
            for key in possible_keys:
                if key in concept_data:
                    content = concept_data[key]
                    used_key = key
                    break
            
            if content is None:
                result = f"Section '{section_name}' not found for concept '{concept}'"
            else:
                # Handle different data types but keep minimal processing
                if isinstance(content, list):
                    result = "\n".join([str(item) for item in content]) if content else ""
                elif isinstance(content, dict):
                    result = json.dumps(content, indent=2)  # Pretty print for better LLM parsing
                else:
                    result = str(content) if content else ""
        
        # 🔍 GROUND TRUTH RETRIEVAL - OUTPUT 🔍
        print("📚 GROUND TRUTH RETRIEVAL - COMPLETED")
        print(f"📋 SECTION_KEY_USED: {used_key if 'used_key' in locals() else section_name}")
        print(f"📏 RESULT_LENGTH: {len(result)} characters")
        print(f"📄 RESULT_PREVIEW: {result[:200]}...")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        error_msg = f"Error retrieving ground truth for {concept} - {section_name}: Error: {e}, Used file: {file_path if 'file_path' in locals() else 'N/A'}"
        print(f"❌ {error_msg}")
        print("=" * 70)
        # return result
        raise RuntimeError(error_msg) from e

# ─────────────────────────────────────────────────────────────────────
# Simulation configuration helpers
# ─────────────────────────────────────────────────────────────────────

def create_simulation_config(variables: List, concept: str, action_config: Optional[Dict] = None) -> Dict:

    action_config = action_config or {}
    # Default parameters
    base_params = {"length": 1.0, "gravity": 9.8, "amplitude": 75, "mass": 1.0}
    
    # Extract independent variable that's being changed
    independent_var = None
    for var in variables:
        # Handle both Pydantic objects (legacy) and dictionaries (new format)
        if hasattr(var, 'role'):  # Pydantic object
            if var.role == "independent":
                independent_var = var.name.lower()
                break
        elif isinstance(var, dict):  # Dictionary format
            if var.get('role') == "independent":
                independent_var = var.get('name', '').lower()
                break
    
    if not independent_var:
        raise ValueError(f"No independent variable found for concept: {concept}")
    
    # Map concept variables to simulation parameters
    if "length" in independent_var or "length" in concept.lower():
        return {
            "concept": concept,
            "parameter_name": "length",
            "before_params": {**base_params, "length": 1.0},
            "after_params": {**base_params, "length": 3.0},
            "action_description": "increasing the pendulum length from 1.0m to 3.0m",
            "timing": {"before_duration": 8, "transition_duration": 3, "after_duration": 8},
            "agent_message": "Watch how the period changes as I increase the length for you...(Before Time Period was 2.01s and After Time Period is 3.47s)"
        }
    elif "gravity" in independent_var or "gravity" in concept.lower():
        return {
            "concept": concept,
            "parameter_name": "gravity",
            "before_params": {**base_params, "gravity": 9.8},
            "after_params": {**base_params, "gravity": 50.0},  # High gravity demonstration
            "action_description": "changing gravity from Earth (9.8 m/s²) to high gravity (50 m/s²)",
            "timing": {"before_duration": 8, "transition_duration": 3, "after_duration": 8},
            "agent_message": "Watch carefully as I change the gravity for you to see how the period changes...(Before Time Period was 2.01s and After Time Period is 0.89s)"
        }
    elif "amplitude" in independent_var or "angle" in independent_var:
        return {
            "concept": concept,
            "parameter_name": "amplitude",
            "before_params": {**base_params, "amplitude": 30},
            "after_params": {**base_params, "amplitude": 60},
            "action_description": "increasing the starting angle from 30° to 60°",
            "timing": {"before_duration": 6, "transition_duration": 2, "after_duration": 6},
            "agent_message": "Watch closely as I increase the swing angle for you to see how the period changes...(The time periods will remaain the same as 2.01 seconds before and after)"
        }
    elif "mass" in independent_var or "bob" in independent_var:
        # For pendulum physics, mass doesn't affect the period, but we can demonstrate this
        return {
            "concept": concept,
            "parameter_name": "mass_demo",
            "before_params": {**base_params, "mass": 1},
            "after_params": {**base_params, "mass": 10},  # Same parameters to show no change
            "action_description": "comparing pendulums with different bob masses (but same period)",
            "timing": {"before_duration": 8, "transition_duration": 3, "after_duration": 8},
            "agent_message": "Watch this carefully! I'll show you how changing the bob mass affects the period - this might surprise you!(The time periods will remain the same as 2.01 seconds before and after)"
        }
    elif "frequency" in independent_var or "period" in independent_var:
        # Demonstrate period/frequency by changing length
        return {
            "concept": concept,
            "parameter_name": "length",
            "before_params": {**base_params, "length": 0.5},
            "after_params": {**base_params, "length": 2.0},
            "action_description": "changing length to show how period and frequency are related",
            "timing": {"before_duration": 7, "transition_duration": 3, "after_duration": 7},
            "agent_message": "I'll show you how changing length affects both period and frequency - watch this demonstration..."
        }
    else:
        raise ValueError(f"Unrecognized independent variable '{independent_var}' for concept: {concept}")


def select_most_relevant_image_for_concept_introduction(concept: str, definition_context: str, model: str = "gemma-3-27b-it") -> Optional[Dict]:
    """
    Select the most pedagogically relevant image for introducing a concept.
    Uses the concept-to-file mapping to find the correct JSON file.
    
    Args:
        concept: The concept name (can be in any case)
        definition_context: The context/definition being provided to the student
        model: Model to use for image selection. Defaults to gemma-3-27b-it.
    
    Returns:
        Dict with url, description, and relevance_reason, or None if no images found
    """
    try:
        # Get concept-to-file mapping
        mapping = _build_concept_to_file_mapping()
        concept_key = concept.lower().strip()
        
        # Find the JSON file for this concept
        json_file_path = mapping.get(concept_key)
        
        if not json_file_path:
            print(f"⚠️ Concept '{concept}' not in mapping, cannot retrieve images")
            return None
        
        print(f"📂 Looking for images in: {json_file_path}")
        
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract concept data
        concept_data = _extract_concept_data_from_json(data, concept)
        
        if not concept_data:
            print(f"Concept '{concept}' not found in JSON data")
            return None
        
        # Get images from concept data (handle different key variations)
        # 8.json and NCERT Class 7.json use "images" (plural)
        # 11.json and 12.json use "image" (singular)
        available_images = concept_data.get("images", concept_data.get("image", concept_data.get("Images", [])))
        
        if not available_images:
            print(f"No images found for concept '{concept}'")
            return None
        
        # Create LLM prompt for image selection
        images_text = "\n".join([
            f"Image {i+1}: {img.get('description', 'No description')}" 
            for i, img in enumerate(available_images)
        ])
        
        selection_prompt = f"""You are helping select the most pedagogically effective image for introducing the concept "{concept}" to a Class 7 student.

Context being provided to student:
{definition_context}

Available images:
{images_text}

Select the image that would be MOST helpful for a 12-13 year old student to understand this concept during the definition phase.

Consider:
- Visual clarity and simplicity
- Direct relevance to the core concept
- Age-appropriate complexity
- Ability to reinforce the definition

Respond with JSON only:
{{
    "selected_image_number": <1-based index>,
    "relevance_reason": "<2-3 sentences explaining why this image is best for concept introduction>"
}}"""
        
        # Get LLM response with fallback
        response = invoke_llm_with_fallback(
            [HumanMessage(content=selection_prompt)],
            operation_name="Image selection",
            model=model
        )
        
        # Parse response
        json_text = extract_json_block(response.content)
        selection_data = json.loads(json_text)
        
        selected_index = selection_data.get("selected_image_number", 1) - 1  # Convert to 0-based
        
        if 0 <= selected_index < len(available_images):
            selected_image = available_images[selected_index]
            return {
                "url": selected_image.get("url", ""),
                "description": selected_image.get("description", ""),
                "relevance_reason": selection_data.get("relevance_reason", "This image was selected as most relevant for concept introduction.")
            }
        else:
            print(f"Invalid image selection index: {selected_index}")
            return None
            
    except Exception as e:
        print(f"Error selecting image for concept '{concept}': {e}")
        import traceback
        traceback.print_exc()
        return None


# ─── Memory Optimization Functions ─────────────────────────────────────────────

def identify_node_segments_from_transitions(messages: list, transitions: list) -> list:
    """
    Split messages into segments based on recorded node transitions.
    Transition happens AFTER the agent response, so messages belong to the 'from_node'.
    """
    if not transitions:
        # No transitions recorded, treat all messages as one segment  
        return [{"node": "unknown", "messages": messages, "start_idx": 0, "end_idx": len(messages)}]
    
    segments = []
    start_idx = 0
    
    for transition in transitions:
        # Messages up to (and including) transition point belong to 'from_node'
        end_idx = transition["transition_after_message_index"] 
        
        if end_idx > start_idx:
            segments.append({
                "node": transition["from_node"],
                "messages": messages[start_idx:end_idx],
                "start_idx": start_idx,
                "end_idx": end_idx
            })
        start_idx = end_idx
    
    # Add the final segment (current node messages) - messages after last transition
    if start_idx < len(messages):
        current_node = transitions[-1]["to_node"] if transitions else "current"
        segments.append({
            "node": current_node,
            "messages": messages[start_idx:], 
            "start_idx": start_idx,
            "end_idx": len(messages)
        })
    
    return segments

def create_educational_summary(messages: list, model: str = "gemma-3-27b-it") -> str:
    """
    Use LLM to create a proper educational summary of the conversation.
    
    Args:
        messages: List of conversation messages
        model: Model to use for summarization. Defaults to gemma-3-27b-it.
    """
    if not messages:
        return ""
    
    # Extract agent messages for summarization
    agent_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
    student_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
    
    if not agent_messages:
        return f"Student made {len(student_messages)} responses"
    
    # Build conversation text for summarization
    conversation_text = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_text += f"Student: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_text += f"Agent: {msg.content}\n"
    
    # Limit conversation text to avoid token overflow
    if len(conversation_text) > 2000:
        conversation_text = conversation_text[:2000] + "..."
    
    # Use LLM to summarize
    summary_prompt = f"""Summarize the following educational conversation in 2-3 sentences, focusing on:
- What concept was being taught
- Student's understanding level
- Key pedagogical interactions

Conversation:
{conversation_text}

Summary:"""
    
    
    try:
        summary_response = invoke_llm_with_fallback(
            [HumanMessage(content=summary_prompt)],
            operation_name="Educational summary",
            model=model
        )
        return summary_response.content.strip()
    except Exception as e:
        print(f"❌ Error creating LLM summary: {e}")
        # Fallback to simple summary if LLM fails
        return f"Educational discussion with {len(messages)} exchanges about the concept"

def create_educational_summary_from_text(conversation_text: str, model: str = "gemma-3-27b-it") -> str:
    """
    Create an LLM-generated summary from conversation text.
    
    Args:
        conversation_text: Text of the conversation to summarize
        model: Model to use for summarization. Defaults to gemma-3-27b-it.
    """
    try:
        if not conversation_text.strip():
            return "Empty conversation segment"
        
        # Limit conversation text to avoid token overflow
        if len(conversation_text) > 2000:
            conversation_text = conversation_text[:2000] + "..."
        
        # Create educational summary prompt
        summary_prompt = f"""Summarize the following educational conversation in 2-3 sentences, focusing on:
- What concept was being taught
- Student's understanding level  
- Key pedagogical interactions

Conversation:
{conversation_text}

Summary:"""

        summary_response = invoke_llm_with_fallback(
            [HumanMessage(content=summary_prompt)],
            operation_name="Summary from text",
            model=model
        )
        return summary_response.content.strip()
    except Exception as e:
        print(f"❌ Error creating LLM summary: {e}")
        # Fallback to simple summary if LLM fails
        return "Educational discussion about the concept"

def build_node_aware_conversation_history(state: AgentState, current_node: str) -> str:
    """
    Keep exact messages from current and previous node interactions.
    Use cached summaries and only summarize new content incrementally.
    """
    messages = state.get("messages", [])
    transitions = state.get("node_transitions", [])
    model = state.get("model", "gemma-3-27b-it")
    
    # For short conversations, use full history
    if len(messages) <= 6:
        return build_conversation_history(state)
    
    # Get node segments based on recorded transitions
    segments = identify_node_segments_from_transitions(messages, transitions)
    
    print(f"📊 MEMORY OPTIMIZATION: Found {len(segments)} node segments")
    
    if len(segments) >= 2:
        # Keep current + previous node segments exact
        current_segment = segments[-1]  # Current node
        previous_segment = segments[-2]  # Previous node
        older_segments = segments[:-2]   # Everything before previous node
        
        print(f"📊 Current node: {current_segment['node']} ({len(current_segment['messages'])} messages)")
        print(f"📊 Previous node: {previous_segment['node']} ({len(previous_segment['messages'])} messages)")
        print(f"📊 Older segments: {len(older_segments)} segments")
        
        # Handle summary efficiently
        summary = ""
        
        if older_segments:
            # Calculate what needs to be summarized
            older_messages = []
            for segment in older_segments:
                older_messages.extend(segment["messages"])
            
            # Get last older index directly from segment metadata (O(1) operation)
            last_older_index = older_segments[-1]["end_idx"] - 1 if older_segments else -1
            
            # Check if we need to update summary
            if last_older_index <= state.get("summary_last_index", -1):
                # Use existing summary - no new messages to summarize
                summary = state.get("summary", "")
                print(f"📊 ✅ Using existing summary (covers up to index {state.get('summary_last_index', -1)})")
            else:
                # Need to update summary with new messages
                new_messages_start = state.get("summary_last_index", 0) + 1
                new_messages = messages[new_messages_start:last_older_index + 1]

                if state.get("summary", ""):
                    # Combine old summary with new messages
                    print(f"Old Summary: {state.get('summary')}")
                    combined_content = f"Previous summary: {state.get('summary')}\n\nNew messages:\n"
                    for msg in new_messages:
                        if isinstance(msg, HumanMessage):
                            combined_content += f"Student: {msg.content}\n"
                        elif isinstance(msg, AIMessage):
                            combined_content += f"Agent: {msg.content}\n"
                    
                    print(f"📊 🔄 Updating summary: old summary + {len(new_messages)} new messages...")
                    summary = create_educational_summary_from_text(combined_content, model=model)
                else:
                    # First time - just summarize the messages
                    print(f"📊 🔄 Creating first summary for {len(new_messages)} messages...")
                    summary = create_educational_summary(new_messages, model=model)
                
                # Update summary state
                state["summary"] = summary
                state["summary_last_index"] = last_older_index
                print(f"📊 💾 Updated summary (now covers up to index {last_older_index})")
            
            summary = f"Previous conversation summary: {summary}\n\n"
        
        # Format recent messages (previous + current node) exactly
        recent_messages = previous_segment["messages"] + current_segment["messages"]
        recent_text = ""
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                recent_text += f"Student: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_text += f"Agent: {msg.content}\n"
        
        optimized_history = summary + recent_text.strip()
        print(f"📊 OPTIMIZATION RESULT: {len(build_conversation_history(state))} -> {len(optimized_history)} chars")
        return optimized_history
    
    else:
        # Not enough transitions, fall back to regular history
        print(f"📊 Not enough transitions, using full history")
        return build_conversation_history(state)

def reset_memory_summary(state: AgentState):
    """
    Reset the memory summary. Useful for testing or manual management.
    """
    if "summary" in state:
        del state["summary"]
        del state["summary_last_index"]
        print("📊 🗑️ Memory summary reset")

# ─── Pedagogical‐move context (shared between traditional and simulation nodes) ───────────

