import json
from typing import Literal, Optional, Dict

from pydantic import BaseModel, Field, field_validator
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser
# from langchain_.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import shared utilities
from utils.shared_utils import (
    AgentState,
    extract_json_block,
    get_llm,
    llm_with_history,
    build_conversation_history,
    build_prompt_from_template,
    build_prompt_from_template_optimized,
    get_ground_truth_from_json,
    select_most_relevant_image_for_concept_introduction,
    add_ai_message_to_conversation,
    add_system_message_to_conversation,
    # Import autosuggestion pool constants (single source of truth)
    POSITIVE_POOL,
    NEGATIVE_POOL,
    SPECIAL_HANDLING_POOL,
)

PEDAGOGICAL_MOVES: Dict[str, Dict[str, str]] = {
    "APK": {
        "goal": "Activate prior knowledge; pose a hook linking the concept to everyday intuition.",
        "constraints": "Do not reveal definitions or answers; question must be common-sense answerable."
    },  
    "CI": {
        "goal": "Provide a concise definition (≤30 words); ask learner to restate it.",
        "constraints": "Keep definition crisp; hint why it matters in ≤1 phrase."
    },
    "GE": {
        "goal": "Ask a why/how question to explore the mechanism; respond with hint or affirmation.",
        "constraints": "Provide only one nudge if learner struggles; do not lecture."
    },
    "MH": {
        "goal": "Detect and correct misconceptions gently.",
        "constraints": "Start positive; keep correction ≤2 sentences."
    },
    "AR": {
        "goal": "Generate a short quiz (T/F, MCQ, or short answer) and prompt learner.",
        "constraints": "Give immediate feedback after each question."
    },
    "TC": {
        "goal": "Pose a hypothetical transfer question applying the concept in a new context.",
        "constraints": "Scenario plausible but unfamiliar; ≤2 sentences."
    },
    "RLC": {
        "goal": "Provide a real-life application/context; ask if learner has seen or used it themselves.",
        "constraints": "Story ≤3 sentences; open-ended question."
    },
    "END": {
        "goal": "Summarize 2–3 bullet takeaways; offer next actions.",
        "constraints": "Bullet format; no new content."
    },
}

# ─── Helper function for combining autosuggestions ────────────────────────────
# Note: Autosuggestion pool constants (POSITIVE_POOL, NEGATIVE_POOL, SPECIAL_HANDLING_POOL)
# are imported from utils.shared_utils - single source of truth for all pool definitions

def combine_autosuggestions(parsed_response: dict, fallback_suggestions: list[str]) -> tuple[list[str], dict]:
    """
    Combine all 4 autosuggestion types (positive, negative, special, dynamic) into final list.
    Handles None values when agent asks questions or says "let me think".
    
    Args:
        parsed_response: Parsed LLM response dict
        fallback_suggestions: Default suggestions if LLM didn't provide valid ones
    
    Returns:
        Tuple of (final_suggestions_list, selections_dict)
    """
    # Extract all four autosuggestions (handle None values)
    positive = parsed_response.get('positive_autosuggestion')
    if positive and isinstance(positive, str):
        positive = positive.strip()
    
    negative = parsed_response.get('negative_autosuggestion', "").strip()
    special = parsed_response.get('special_handling_autosuggestion', "").strip()
    
    dynamic = parsed_response.get('dynamic_autosuggestion')
    if dynamic and isinstance(dynamic, str):
        dynamic = dynamic.strip()
    
    # Combine into ordered list, filtering out None and empty strings
    print("Positive:", positive)
    print("Negative:", negative)
    print("Special:", special)
    print("Dynamic:", dynamic)
    final_suggestions = [s for s in [positive, negative, special, dynamic] if s]
    print("Final Suggestions before validation:", final_suggestions)
    
    # Check if we have at least negative and special (minimum required when asking questions)
    # When positive/dynamic are None (question scenario), we should have 2 suggestions
    is_question_scenario = positive is None or dynamic is None
    expected_count = 2 if is_question_scenario else 4
    
    if len(final_suggestions) < expected_count:
        print(f"⚠️ WARNING: Expected {expected_count} suggestions, got {len(final_suggestions)}. Using fallback.")
        final_suggestions = fallback_suggestions
        # Create fallback dict
        selections_dict = {
            'positive': fallback_suggestions[0] if len(fallback_suggestions) > 0 else "",
            'negative': fallback_suggestions[1] if len(fallback_suggestions) > 1 else "",
            'special': fallback_suggestions[2] if len(fallback_suggestions) > 2 else "",
            'dynamic': fallback_suggestions[3] if len(fallback_suggestions) > 3 else ""
        }
    else:
        # Create selections dict for state storage (preserve None values)
        selections_dict = {
            'positive': positive,
            'negative': negative,
            'special': special,
            'dynamic': dynamic
        }
        print(f"✅ Combined 4 autosuggestions: positive='{positive}', negative='{negative}', special='{special}', dynamic='{dynamic[:30] if dynamic else 'None'}...'")
    
    return final_suggestions, selections_dict


# ─── Pydantic response models ─────────────────────────────────────────────────

class BaseAutosuggestionResponse(BaseModel):
    """Base model with shared autosuggestion fields using Literal for validation.
    
    All pedagogical response models inherit from this to ensure consistent
    autosuggestion behavior across the system.
    
    Requires exactly 4 autosuggestions:
    1. One positive (affirmation/understanding)
    2. One negative (confusion/uncertainty)
    3. One special handling (triggers hint/example/simpler explanation)
    4. One dynamic (contextually generated exploration prompt)
    """
    
    # Positive pool selection using Literal - synced with POSITIVE_POOL
    # None allowed when agent asks a question
    positive_autosuggestion: Literal[
        "I understand, continue",
        "Yes, got it",
        "That makes sense",
        "Let's proceed further",
        "I'm following along",
        None
    ] = Field(
        default=None,
        description="Select EXACTLY ONE positive/affirmative suggestion that shows understanding or agreement. Use None (null in JSON) if your feedback contains a direct question to the student."
    )
    
    # Negative pool selection using Literal - synced with NEGATIVE_POOL
    negative_autosuggestion: Literal[
        "I'm not sure",
        "I don't know",
        "I'm confused",
        "Not very clear",
        "Can you explain differently?"
    ] = Field(
        description="Select EXACTLY ONE negative/uncertain suggestion that expresses confusion or need for help"
    )
    
    # Special handling pool selection using Literal - synced with SPECIAL_HANDLING_POOL
    special_handling_autosuggestion: Literal[
        "Can you give me a hint?",
        "Can you explain that simpler?",
        "Give me an example"
    ] = Field(
        description="Select EXACTLY ONE special handling suggestion that will trigger pedagogical intervention"
    )
    
    # Dynamic suggestion - str or None (can't use Literal for open string values)
    dynamic_autosuggestion: str | None = Field(
        default=None,
        description="""Generate ONE unique exploratory autosuggestion (12-15 words max) that:
        - Is contextually relevant to the current conversation and your message
        - Points to a specific unexplored aspect related to what you just explained
        - Should nudge student to think about a related concept/application/implication
        - Uses short noun-phrase or statement format that evokes curiosity
        - Adjusts depth based on student_level:
          * low: concrete, visible aspects (where it happens, what is used/made)
          * medium: cause-effect, constraints (why needed, what enables/prevents)
          * advanced: dependencies, variations, implications (how changes affect, limiting factors)
        - Must be DIFFERENT from all pool suggestions above
        - Use None (null in JSON) if your feedback contains a direct question to the student or if you say 'let me think'"""
    )
    
    @field_validator('dynamic_autosuggestion')
    @classmethod
    def validate_dynamic_not_empty(cls, v):
        """Ensure dynamic autosuggestion is provided and not just whitespace, or is None."""
        if v is None:
            return None
        if not v.strip():
            raise ValueError("Dynamic autosuggestion must be provided and cannot be empty (or use None if asking a question).")
        return v.strip()
    
    @field_validator('positive_autosuggestion')
    @classmethod
    def validate_positive_in_pool(cls, v):
        """Validate positive selection matches pool (runtime check for sync)."""
        if v is None:
            return None
        if v not in POSITIVE_POOL:
            raise ValueError(f"positive_autosuggestion must be from POSITIVE_POOL or None. Got: {v}")
        return v
    
    @field_validator('negative_autosuggestion')
    @classmethod
    def validate_negative_in_pool(cls, v):
        """Validate negative selection matches pool (runtime check for sync)."""
        if v not in NEGATIVE_POOL:
            raise ValueError(f"negative_autosuggestion must be from NEGATIVE_POOL. Got: {v}")
        return v
    
    @field_validator('special_handling_autosuggestion')
    @classmethod
    def validate_special_in_pool(cls, v):
        """Validate special handling selection matches pool (runtime check for sync)."""
        if v not in SPECIAL_HANDLING_POOL:
            raise ValueError(f"special_handling_autosuggestion must be from SPECIAL_HANDLING_POOL. Got: {v}")
        return v


class ApkResponse(BaseAutosuggestionResponse):
    feedback: str
    next_state: Literal["CI", "APK"]

class CiResponse(BaseAutosuggestionResponse):
    feedback: str
    next_state: Literal["CI","SIM_CC"]

class GeResponse(BaseAutosuggestionResponse):
    feedback: str
    next_state: Literal["AR", "GE"]
    # correction: Optional[str] = None  # OLD: Used when routing to SIM_VARS/MH

class MhResponse(BaseModel):
    feedback: str
    next_state: Literal["MH", "AR"]

class ArResponse(BaseAutosuggestionResponse):
    score: float
    feedback: str
    next_state: Literal["GE", "TC"]

class TcResponse(BaseAutosuggestionResponse):
    correct: bool
    feedback: str

class RlcResponse(BaseAutosuggestionResponse):
    feedback: str
    next_state: Literal["RLC", "END"]

# ─── Parsers ───────────────────────────────────────────────────────────────────

apk_parser = PydanticOutputParser(pydantic_object=ApkResponse)
ci_parser  = PydanticOutputParser(pydantic_object=CiResponse)
ge_parser  = PydanticOutputParser(pydantic_object=GeResponse)
mh_parser  = PydanticOutputParser(pydantic_object=MhResponse)
ar_parser  = PydanticOutputParser(pydantic_object=ArResponse)
tc_parser  = PydanticOutputParser(pydantic_object=TcResponse)
rlc_parser = PydanticOutputParser(pydantic_object=RlcResponse)


# ─── Handler functions for special autosuggestions ────────────────────────────────

def handle_hint(state: AgentState) -> AgentState:
    """Generate a contextual hint without revealing the answer"""
    current_node = state.get("current_state", "UNKNOWN")
    
    hint_prompt = """The student requested a hint to help them answer the question.

Generate a supportive hint (2-3 sentences) that:
1. Helps the student think through the problem without revealing the full answer
2. References what was just discussed in the conversation
3. Uses tone and encouragement appropriate to the conversation context

Be natural and conversational."""
    
    # Build prompt - conversation history included automatically, no autosuggestions needed
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=hint_prompt,
        state=state,
        include_last_message=False,
        include_instructions=False,
        current_node=current_node,
        include_autosuggestions=False
    )
    
    resp = llm_with_history(state, final_prompt)
    hint_content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
    
    # Update agent output with hint
    state["agent_output"] = hint_content
    
    print("=" * 80)
    print("🔍 HANDLER: HINT GENERATED")
    print("=" * 80)
    print(f"💡 HINT: {hint_content[:100]}...")
    print("=" * 80)
    
    return state


def handle_explain_simpler(state: AgentState) -> AgentState:
    """Rephrase the last explanation in simpler language"""
    current_node = state.get("current_state", "UNKNOWN")
    
    simplify_prompt = """The student asked for a simpler explanation.

Rephrase the last explanation you gave using:
1. Very simple words suitable for a class 7 student
2. Shorter sentences
3. Everyday examples if helpful

Keep the same meaning but make it much easier to understand. Use tone and follow-up appropriate to the conversation context."""
    
    # Build prompt - conversation history included automatically, no autosuggestions needed
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=simplify_prompt,
        state=state,
        include_last_message=False,
        include_instructions=False,
        current_node=current_node,
        include_autosuggestions=False
    )
    
    resp = llm_with_history(state, final_prompt)
    simple_content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
    
    # Update agent output with simplified version
    state["agent_output"] = simple_content
    
    print("=" * 80)
    print("🔍 HANDLER: SIMPLIFIED EXPLANATION")
    print("=" * 80)
    print(f"📝 SIMPLIFIED: {simple_content[:100]}...")
    print("=" * 80)
    
    return state


def handle_example(state: AgentState) -> AgentState:
    """Provide a concrete example to illustrate the concept"""
    concept_title = state.get("concept_title", "")
    current_node = state.get("current_state", "UNKNOWN")
    
    example_prompt = f"""The student asked for an example.

Provide a simple, concrete example related to '{concept_title}' that:
1. Is brief (2-3 sentences)
2. Uses everyday situations a class 7 student can relate to
3. Helps illustrate what was just discussed

Use tone and follow-up appropriate to the conversation context."""
    
    # Build prompt - conversation history included automatically, no autosuggestions needed
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=example_prompt,
        state=state,
        include_last_message=False,
        include_instructions=False,
        current_node=current_node,
        include_autosuggestions=False
    )
    
    resp = llm_with_history(state, final_prompt)
    example_content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
    
    # Update agent output with example
    state["agent_output"] = example_content
    
    print("=" * 80)
    print("🔍 HANDLER: EXAMPLE PROVIDED")
    print("=" * 80)
    print(f"🎯 EXAMPLE: {example_content[:100]}...")
    print("=" * 80)
    
    return state


def handle_dynamic_suggestion(state: AgentState) -> AgentState:
    """Process dynamic autosuggestion based on student level and their specific request."""
    student_level = state.get("student_level", "medium")
    dynamic_request = state.get("last_user_msg", "")
    current_node = state.get("current_state", "UNKNOWN")
    
    # Level-specific context for tailored responses
    level_instructions = {
        "low": "Use very simple language, short sentences, and provide step-by-step guidance. Avoid complex terminology.",
        "medium": "Use clear explanations with moderate complexity. Balance detail with accessibility.",
        "advanced": "Provide deeper insights, encourage critical thinking, and explore nuances of the concept."
    }
    
    instruction = level_instructions.get(student_level, level_instructions["medium"])
    
    dynamic_prompt = f"""The student (ability level: {student_level}) asked: "{dynamic_request}"

Task: Respond to their specific request keeping in mind they are a {student_level}-level student.
{instruction}

Your response should:
1. Be brief (2-3 sentences) and directly address what they asked
2. Reference the ongoing conversation context
3. Use tone and follow-up appropriate to the conversation flow

Be natural and supportive.and follow-up appropriate to the conversation context."""
    
    # Build prompt - conversation history included automatically, no autosuggestions needed
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=dynamic_prompt,
        state=state,
        include_last_message=False,
        include_instructions=False,
        current_node=current_node,
        include_autosuggestions=False
    )
    
    resp = llm_with_history(state, final_prompt)
    dynamic_content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
    
    # Update agent output with level-aware dynamic response
    state["agent_output"] = dynamic_content
    
    print("=" * 80)
    print("🔍 HANDLER: DYNAMIC SUGGESTION PROCESSED")
    print("=" * 80)
    print(f"🎯 STUDENT_LEVEL: {student_level}")
    print(f"💬 REQUEST: {dynamic_request}")
    print(f"📝 RESPONSE: {dynamic_content[:100]}...")
    print("=" * 80)
    
    return state


# def handle_repeat(state: AgentState) -> AgentState:
#     """Repeat the last agent output (restore from backup)"""
#     backup = state.get("last_agent_output_backup", "")
#     
#     if backup:
#         state["agent_output"] = backup
#         print("=" * 80)
#         print("🔍 HANDLER: REPEATED LAST MESSAGE")
#         print("=" * 80)
#     else:
#         # Fallback if no backup available
#         state["agent_output"] = "I apologize, I don't have a previous message to repeat. Let's continue from here."
#         print("=" * 80)
#         print("🔍 HANDLER: NO BACKUP AVAILABLE FOR REPEAT")
#         print("=" * 80)
#     
#     return state


# ─── Autosuggestion Manager Node ──────────────────────────────────────────────

def autosuggestion_manager_node(state: AgentState) -> AgentState:
    """
    Manager node that ONLY handles special handler suggestions.
    Autosuggestions are already set by pedagogical nodes.
    This node runs ONLY when user clicks an autosuggestion button.
    
    CRITICAL: Sets a flag to indicate if handler was triggered.
    Graph routing will use this to decide whether to interrupt.
    """
    
    print("=" * 80)
    print("🎯 AUTOSUGGESTION MANAGER NODE - ENTRY")
    print("=" * 80)
    
    last_user_msg = state.get("last_user_msg", "")
    special_handling = state.get("special_handling_autosuggestion", "")
    dynamic_suggestion = state.get("dynamic_autosuggestion", "")
    
    # Check if user clicked the SPECIAL HANDLING suggestion
    if last_user_msg == special_handling and special_handling:
        print(f"🔧 SPECIAL HANDLING TRIGGERED: {last_user_msg}")
        
        if last_user_msg == "Can you give me a hint?":
            state = handle_hint(state)
        elif last_user_msg == "Can you explain that simpler?":
            state = handle_explain_simpler(state)
        elif last_user_msg == "Give me an example":
            state = handle_example(state)
        
        # Mark that we need to show the handler output to user (interrupt)
        state["handler_triggered"] = True
    
    # Check if user clicked the DYNAMIC suggestion
    elif last_user_msg == dynamic_suggestion and dynamic_suggestion:
        print(f"🎯 DYNAMIC HANDLER TRIGGERED: {last_user_msg}")
        state = handle_dynamic_suggestion(state)
        # Mark that we need to show the handler output to user (interrupt)
        state["handler_triggered"] = True
    
    else:
        print(f"📝 NORMAL FLOW AUTOSUGGESTION: {last_user_msg}")
        print("   (Positive or negative suggestion - flow continues without pause)")
        state["handler_triggered"] = False
    
    # Reset click flag for next interaction
    state["clicked_autosuggestion"] = False
    
    print(f"🔄 PRESERVING current_state: {state.get('current_state', 'UNKNOWN')}")
    print(f"📤 AUTOSUGGESTIONS ALREADY SET: {state.get('autosuggestions', [])}")
    print(f"⏸️ HANDLER TRIGGERED: {state.get('handler_triggered', False)}")
    print("=" * 80)
    
    # CRITICAL: Preserve current_state - do NOT modify pedagogical routing
    return state


# ─── Node definitions ───────────────────────────────────────────────────────────

def start_node(state: AgentState) -> AgentState:
    # Base system prompt
    system_prompt = f"""
        You are an educational agent helping a learner understand '{state["concept_title"]}'. The learner is a student of class 7. Remember that you are interacting directly with the learner.\n"
        "Greet the learner and ask if they are ready to begin."
        "DONT use emojis as a TTS to speech model will break because of that."
    """
    
    # Conditionally add Kannada instruction if is_kannada is True
    if state.get("is_kannada", False):
        system_prompt += "\nAlso Remember that the student is of Kannada origin and understands only kannada. So speak to the student in kannada. The script has to be kannada and not english.Show text in kannada only."
    
    # Build final prompt using optimized template
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=False,
        include_instructions=False,
        current_node="START"
    )
    
    print("IN START NODE")
    resp = llm_with_history(state, final_prompt)  # Using regular llm_with_history since prompt is pre-built
    # Apply JSON extraction in case LLM wraps response in markdown
    content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
    
    # 🔍 START NODE - CONTENT PROCESSING 🔍
    print("=" * 80)
    print("🎯 START NODE - CONTENT OUTPUT 🎯")
    print("=" * 80)
    print(f"📄 CONTENT: {content}")
    print(f"📏 CONTENT_LENGTH: {len(content)} characters")
    print(f"📊 CONTENT_TYPE: {type(content).__name__}")
    print(f"🔧 USED_JSON_EXTRACTION: {resp.content.strip().startswith('```')}")
    print("=" * 80)
    
    # Initialize state fields to prevent KeyErrors
    state["agent_output"]  = content
    state["current_state"] = "APK"
    state["messages"] = []  # Initialize empty message list
    add_system_message_to_conversation(state, content)
    state["summary"] = ""  # Initialize summary
    state["summary_last_index"] = -1  # Initialize summary index
    return state

def apk_node(state: AgentState) -> AgentState:
    if not state.get("asked_apk", False):
        state["asked_apk"] = True
        state["apk_tries"] = 0
        # Include ground truth for Concept Definition
        gt = get_ground_truth_from_json(state["concept_title"], "Concept Definition")
        system_prompt = f"""
            Please use the following ground truth as a baseline and build upon it, but do not deviate too much.\n
            Ground truth (Concept Definition):\n{gt}\nGenerate one hook question that activates prior knowledge for '{state["concept_title"]}'.,
            Remember you are talking directly to the students so only output the hook question and nothing else.
        """
        
        # Build final prompt using optimized template
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=system_prompt,
            state=state,
            include_last_message=False,
            include_instructions=False,
            current_node="APK"
        )
            
        resp = llm_with_history(state, final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
        
        # 🔍 APK NODE - FIRST PASS CONTENT 🔍
        print("=" * 80)
        print("🎯 APK NODE - FIRST PASS CONTENT OUTPUT 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔧 USED_JSON_EXTRACTION: {resp.content.strip().startswith('```')}")
        print("=" * 80)
        
        state["agent_output"] = content
        add_ai_message_to_conversation(state, content)
        return state

    # Handle student's response after hook question
    state["apk_tries"] = state.get("apk_tries", 0) + 1
    
    # Check if we've reached max tries (5) - provide answer and move to CI
    if state["apk_tries"] >= 5:
        gt = get_ground_truth_from_json(state["concept_title"], "Concept Definition")
        final_system_prompt = f"""Current node: APK (Activate Prior Knowledge) - FINAL ATTEMPT
This is the final attempt to help the student identify the concept.

The student has had 5 attempts to identify '{state["concept_title"]}' but hasn't gotten it right.

Ground truth (Concept Definition): {gt}

Task: Provide the correct identification of '{state["concept_title"]}' in a supportive way that:
1. Acknowledges their effort
2. Gives the correct answer clearly
3. Briefly explains why this is the concept
4. Transitions positively to learning more about it

Respond ONLY with a clear, encouraging message (not JSON - just the message text)."""

        # Build final prompt for revealing the concept
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=final_system_prompt,
            state=state,
            include_last_message=True,
            include_instructions=False,
            parser=None,
            current_node="APK"
        )
        
        final_response = llm_with_history(state, final_prompt).content.strip()
        
        # 🔍 APK NODE - MAX TRIES REACHED 🔍
        print("=" * 80)
        print("🎯 APK NODE - MAX TRIES REACHED, PROVIDING ANSWER 🎯")
        print("=" * 80)
        print(f"🔢 APK_TRIES: {state['apk_tries']}")
        print(f"💬 LLM_FINAL_MESSAGE: {final_response}")
        print("=" * 80)
        
        state["agent_output"] = final_response
        state["current_state"] = "CI"
        add_ai_message_to_conversation(state, final_response)
        return state

    context = json.dumps(PEDAGOGICAL_MOVES["APK"], indent=2)
    system_prompt = f"""Current node: APK (Activate Prior Knowledge)
Possible next_state values:
- "CI": when the student's reply shows they correctly identified '{state["concept_title"]}'.
- "APK": when the student's reply does not clearly identify '{state["concept_title"]}'.

Pedagogical context:
{context}

This is attempt {state["apk_tries"]} of 2 for prior knowledge activation.

Task: Evaluate whether the student identified the concept correctly. Respond ONLY with JSON matching the schema above. If not, help the student to do so.
Remember to give feedback as mentioned in the required schema."""

    # Build final prompt using optimized template with instructions at the end
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=apk_parser,
        current_node="APK",
        include_autosuggestions=True
    )
    
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    try:
        parsed_obj: ApkResponse = apk_parser.parse(json_text)
        parsed = parsed_obj.model_dump()

        # 🔍 APK PARSING OUTPUT - MAIN CONTENT 🔍
        print("=" * 80)
        print("🎯 APK NODE - PARSED OUTPUT CONTENTS 🎯")
        print("=" * 80)
        print(f"📝 FEEDBACK: {parsed['feedback']}")
        print(f"🚀 NEXT_STATE: {parsed['next_state']}")
        print(f"📊 PARSED_TYPE: {type(parsed).__name__}")
        print("=" * 80)

        state["agent_output"]  = parsed['feedback']
        state["current_state"] = parsed['next_state']
        
        # Combine all 4 autosuggestions (positive, negative, special, dynamic)
        final_suggestions, selections = combine_autosuggestions(
            parsed, 
            ["I understand, continue", "I'm not sure", "Can you give me a hint?", "Continue exploring"]
        )
        
        # Store all 4 individual selections
        state["positive_autosuggestion"] = selections['positive']
        state["negative_autosuggestion"] = selections['negative']
        state["special_handling_autosuggestion"] = selections['special']
        state["dynamic_autosuggestion"] = selections['dynamic']
        state["autosuggestions"] = final_suggestions
        
        add_ai_message_to_conversation(state, parsed['feedback'])
    except Exception as e:
        print(f"Error parsing APK response: {e}")
        print(f"Raw response: {raw}")
        print(f"Extracted JSON text: {json_text}")
        raise
    return state

def ci_node(state: AgentState) -> dict:
    # print("REACHED HERE")
    if not state.get("asked_ci", False):
        # Include ground truth for Explanation (with analogies)
        gt = get_ground_truth_from_json(state["concept_title"], "Explanation (with analogies)")
        system_prompt = f"""
            Please use the following ground truth as a baseline and build upon it, but do not deviate too much.
            Ground truth (Explanation):
{gt}
Provide a concise definition (≤30 words) of '{state["concept_title"]}', then ask the learner to restate it.
        """
        
        # Build final prompt using optimized template
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=system_prompt,
            state=state,
            include_last_message=False,
            include_instructions=False,
            current_node="CI"
        )
            
        resp = llm_with_history(state, final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
        
        # NEW: Select most relevant image for concept introduction
        selected_image = select_most_relevant_image_for_concept_introduction(
            concept=state["concept_title"],
            definition_context=gt + "\n\n" + content
        )
        
        # 🔍 CI NODE - FIRST PASS CONTENT 🔍
        print("=" * 80)
        print("🎯 CI NODE - FIRST PASS CONTENT OUTPUT 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔧 USED_JSON_EXTRACTION: {resp.content.strip().startswith('```')}")
        print(f"🖼️ SELECTED_IMAGE: {selected_image['url'] if selected_image else 'None'}")
        print("=" * 80)
        
        # Add AI message to conversation
        add_ai_message_to_conversation(state, content)
    
        
        result = {
            "asked_ci": True,
            "ci_tries": 0,
            "agent_output": content
        }
        
        # Add image metadata if image was selected
        if selected_image:
            result["enhanced_message_metadata"] = {
                "image": selected_image,
                "node": "CI"
            }
        
        return result

    # Increment attempt counter
    ci_tries = state.get("ci_tries", 0) + 1
    
    # Check if we've reached 2 attempts - if so, provide definition and move on
    if ci_tries >= 2:
        # gt = get_ground_truth_from_json(state["concept_title"], "Explanation (with analogies)")
        system_prompt = f"""
            The student has struggled with restating the definition. Provide the correct definition of '{state["concept_title"]}' 
            clearly and encourage them that it's okay to struggle with new concepts. 
            Then say 'Now let's explore this concept deeper with a question.'
        """
        
        # Build final prompt using optimized template
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=system_prompt,
            state=state,
            include_last_message=False,
            include_instructions=False,
            current_node="CI"
        )
            
        resp = llm_with_history(state, final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
        
        # 🔍 CI NODE - AUTO-PROGRESS CONTENT 🔍
        print("=" * 80)
        print("🎯 CI NODE - AUTO-PROGRESS AFTER 2 TRIES 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔢 CI_TRIES: {ci_tries}")
        print("=" * 80)
        
        # Add AI message to conversation before returning
        add_ai_message_to_conversation(state, content)
        
        # Return only the changed keys following LangGraph best practices
        return {
            "ci_tries": ci_tries,
            "agent_output": content,
            "current_state": "SIM_CC",
            "enhanced_message_metadata": {}
        }

    context = json.dumps(PEDAGOGICAL_MOVES["CI"], indent=2)
    system_prompt = f"""Current node: CI (Concept Introduction)
Possible next_state values:
- "SIM_CC": when the student's paraphrase accurately captures the definition and we need to identify key concepts for exploration.
- "CI": when the paraphrase is inaccurate or incomplete.

Pedagogical context:
{context}

This is attempt {state["ci_tries"]} for the student. If they get it wrong this time, we'll provide the correct definition and move on.

Task: Determine if the restatement is accurate. If accurate, move to SIM_CC to identify concepts for exploration. Respond ONLY with JSON matching the schema above. If not, help the student to do so."""

    # Build final prompt using optimized template with instructions at the end
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=ci_parser,
        current_node="CI",
        include_autosuggestions=True
    )
    
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    try:
        parsed_obj: CiResponse = ci_parser.parse(json_text)
        parsed = parsed_obj.model_dump()

        # 🔍 CI PARSING OUTPUT - MAIN CONTENT 🔍
        print("=" * 80)
        print("🎯 CI NODE - PARSED OUTPUT CONTENTS 🎯")
        print("=" * 80)
        print(f"🚀 NEXT_STATE: {parsed['next_state']}")
        print(f"📝 FEEDBACK: {parsed['feedback']}")
        print(f"🔢 CI_TRIES: {ci_tries}")
        print(f"📊 PARSED_TYPE: {type(parsed).__name__}")
        print("=" * 80)

        # Combine all 4 autosuggestions (positive, negative, special, dynamic)
        final_suggestions, selections = combine_autosuggestions(
            parsed,
            ["I understand, continue", "I'm not sure", "Can you give me a hint?", "Continue exploring"]
        )
        
        # Add AI message to conversation before returning
        add_ai_message_to_conversation(state, parsed['feedback'])
        
        # Return only the changed keys following LangGraph best practices
        return {
            "ci_tries": ci_tries,
            "agent_output": parsed['feedback'],
            "current_state": parsed['next_state'],
            "enhanced_message_metadata": {},
            "positive_autosuggestion": selections['positive'],
            "negative_autosuggestion": selections['negative'],
            "special_handling_autosuggestion": selections['special'],
            "dynamic_autosuggestion": selections['dynamic'],
            "autosuggestions": final_suggestions
        }
    except Exception as e:
        print(f"Error parsing CI response: {e}")
        print(f"Raw response: {raw}")
        print(f"Extracted JSON text: {json_text}")
        raise

def ge_node(state: AgentState) -> AgentState:
    # Check if we're coming from AR after finishing a concept
    if state.get("in_simulation", False):
        state["in_simulation"] = False
        
    # Move to next concept if current concept is done
    current_idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    
    if not state.get("asked_ge", False):
        state["asked_ge"] = True
        state["ge_tries"] = 0  # Initialize tries counter
        
        # Check if we have concepts to explore
        if concepts and current_idx < len(concepts):
            current_concept = concepts[current_idx]
            # Include ground truth for Details (facts, sub-concepts)
            gt = get_ground_truth_from_json(state["concept_title"], "Details (facts, sub-concepts)")
            system_prompt = f"""
                Please use the following ground truth as a baseline and build upon it, but do not deviate too much.
                Ground truth (Details):
                {gt}
                We are now exploring concept {current_idx + 1} of {len(concepts)}: '{current_concept}'.
                Generate one 'why' or 'how' question to explore the mechanism of this specific concept within '{state["concept_title"]}'.
            """
        else:
            print("List of concepts:", concepts)
            print("No concepts available for GE node.")
            raise IndexError("No concepts available for exploration.")

        # Build final prompt using optimized template
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=system_prompt,
            state=state,
            include_last_message=False,
            include_instructions=False,
            current_node="GE"
        )
            
        resp = llm_with_history(state, final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
        
        # 🔍 GE NODE - FIRST PASS CONTENT 🔍
        print("=" * 80)
        print("🎯 GE NODE - FIRST PASS CONTENT OUTPUT 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔧 USED_JSON_EXTRACTION: {resp.content.strip().startswith('```')}")
        print(f"🔢 CURRENT_CONCEPT_IDX: {current_idx}")
        print(f"📋 CURRENT_CONCEPT: {concepts[current_idx] if concepts and current_idx < len(concepts) else 'None'}")
        print("=" * 80)
        
        # Add AI message to conversation before returning
        add_ai_message_to_conversation(state, content)
        
        return {
            "asked_ge": True,
            "ge_tries": 0,
            "agent_output": content
        }

    # Handle tries for GE node - increment counter
    state["ge_tries"] = state.get("ge_tries",0) + 1
    
    # Check if we've reached max tries (4) - transition smoothly to AR
    if state["ge_tries"] >= 4:
        # NEW: Let LLM generate a natural transition to AR with gentle clarification
        current_idx = state.get("sim_current_idx", 0)
        concepts = state.get("sim_concepts", [])
        
        if concepts and current_idx < len(concepts):
            current_concept = concepts[current_idx]
            gt_context = get_ground_truth_from_json(state["concept_title"], "Details (facts, sub-concepts)")
            transition_prompt = f"""The student has tried once to explore concept '{current_concept}' within '{state["concept_title"]}'. 
            
Based on their response, provide a gentle clarification or correction to help them understand better. Use this ground truth as reference: {gt_context[:200]}...

Keep your response conversational and supportive. Address any confusion while guiding them toward the correct understanding.

Then transition to testing their understanding by saying something like: 'Now let's see how well you understand this concept with a quick question.'"""
        else:
            gt_context = get_ground_truth_from_json(state["concept_title"], "Details (facts, sub-concepts)")
            transition_prompt = f"""The student has tried once to explore '{state["concept_title"]}'. 
            
Based on their response, provide a gentle clarification or correction to help them understand better. Use this ground truth as reference: {gt_context[:200]}...

Keep your response conversational and supportive. Address any confusion while guiding them toward the correct understanding.

Then transition to testing their understanding by saying something like: 'Now let's see how well you understand this concept with a quick question.'"""
        
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=transition_prompt,
            state=state,
            include_last_message=True,
            include_instructions=False
        )
        
        resp = llm_with_history(state, final_prompt)
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
        
        # Add AI message to conversation before returning
        add_ai_message_to_conversation(state, content)
        
        # Return only the changed keys following LangGraph best practices
        return {
            "agent_output": content,
            "current_state": "AR"  # NEW: Transition directly to AR for assessment
        }
        
        # OLD: Transition to SIM_VARS for simulation-based misconception handling
        # return {
        #     "agent_output": content,
        #     "current_state": "SIM_VARS"  # OLD: Transition to SIM_VARS for proper misconception handling
        # }

    context = json.dumps(PEDAGOGICAL_MOVES["GE"], indent=2)
    current_idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    
    system_prompt = f"""Current node: GE (Guided Exploration)

Current status: 
- Concept {current_idx + 1} of {len(concepts) if concepts else 0}
- Concept name: {concepts[current_idx] if concepts and current_idx < len(concepts) else 'Unknown'}

Possible next_state values:
- "AR": if the student shows understanding and is ready to be assessed on this concept.
- "GE": if you need to ask another question about the same concept.

Choose ONLY from these options

Pedagogical context:
{context}

Task: Evaluate student understanding and decide if they're ready for assessment (AR) or need more exploration (GE). RESPOND ONLY WITH JSON matching the schema above.

OLD OPTIONS (commented out):
# - "SIM_VARS": if you detect a misconception in the student's reasoning (must include a non-empty "correction" ≤2 sentences).
# Task: Detect misconception, correct reasoning, or need for further exploration."""

    # Build final prompt using optimized template with instructions at the end
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=ge_parser,
        current_node="GE",
        include_autosuggestions=True
    )
    
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    try:
        parsed_obj: GeResponse = ge_parser.parse(json_text)
        parsed = parsed_obj.model_dump()  # Convert to dictionary for serialization safety

        # 🔍 GE PARSING OUTPUT - MAIN CONTENT 🔍
        print("=" * 80)
        print("🎯 GE NODE - PARSED OUTPUT CONTENTS 🎯")
        print("=" * 80)
        print(f"📝 FEEDBACK: {parsed['feedback']}")
        print(f"🚀 NEXT_STATE: {parsed['next_state']}")
        print(f"🔧 CORRECTION: {parsed.get('correction')}")
        print(f"📊 PARSED_TYPE: {type(parsed).__name__}")
        print(f"🔢 CURRENT_CONCEPT_IDX: {current_idx}")
        print("=" * 80)

        # Combine all 4 autosuggestions (positive, negative, special, dynamic)
        final_suggestions, selections = combine_autosuggestions(
            parsed,
            ["I understand, continue", "I'm not sure", "Can you give me a hint?", "Continue exploring"]
        )
        
        # Add AI message to conversation before returning
        add_ai_message_to_conversation(state, parsed['feedback'])
        
        update = {
            "agent_output": parsed['feedback'],
            "current_state": parsed['next_state'],
            "positive_autosuggestion": selections['positive'],
            "negative_autosuggestion": selections['negative'],
            "special_handling_autosuggestion": selections['special'],
            "dynamic_autosuggestion": selections['dynamic'],
            "autosuggestions": final_suggestions
        }

        # NEW: Since GE now only goes to AR or GE, no special handling needed
        # Just return the update with the next_state as determined by the LLM
        
        # OLD: Special handling for MH and SIM_VARS transitions
        # if parsed['next_state'] == "MH":
        #     update["last_correction"] = parsed.get('correction') or "Let me clarify that for you."
        # elif parsed['next_state'] == "SIM_VARS":
        #     update["in_simulation"] = True

        return update
    except Exception as e:
        print(f"Error parsing GE response: {e}")
        print(f"Raw response: {raw}")
        print(f"Extracted JSON text: {json_text}")
        raise

def mh_node(state: AgentState) -> AgentState:
    """Misconception Handling node - addresses student misconceptions and handles follow-up questions"""
    
    # First time entering MH: provide the correction from GE node
    if not state.get("asked_mh", False):
        state["asked_mh"] = True
        state["mh_tries"] = 0
        
        # Get the correction from GE node
        correction = state.get("last_correction", "Let me clarify that for you.")
        
        # Provide the correction to the student
        correction_message = f"I understand your thinking, but let me clarify: {correction}"
        
        # 🔍 MH NODE - FIRST PASS CORRECTION 🔍
        print("=" * 80)
        print("🎯 MH NODE - INITIAL CORRECTION PROVIDED 🎯")
        print("=" * 80)
        print(f"📝 CORRECTION: {correction}")
        print(f"💬 MESSAGE: {correction_message}")
        print("=" * 80)
        
        state["agent_output"] = correction_message
        add_ai_message_to_conversation(state, correction_message)
        return state
    
    # Handle student's response after correction
    state["mh_tries"] = state.get("mh_tries", 0) + 1
    
    # Check if we've reached max tries - use LLM for final conclusion
    if state["mh_tries"] >= 2:
        context = json.dumps(PEDAGOGICAL_MOVES["MH"], indent=2)
        correction = state.get("last_correction", "the previous correction")
        
        final_system_prompt = f"""Current node: MH (Misconception Handling) - FINAL ATTEMPT
This is the final attempt to address the student's misconception.

Pedagogical context:
{context}

Original correction provided: {correction}

The student has had 2 attempts to understand the misconception correction, but may still be confused.

Task: Provide a FINAL, CLEAR explanation that:
1. Acknowledges their confusion/persistence 
2. Gives the correct concept one more time in the simplest terms
3. Concludes the misconception handling positively
4. Transitions to moving forward with assessment

Be encouraging but definitive. This is the final clarification before moving to assessment.
Respond ONLY with a clear, conclusive message (not JSON - just the message text)."""

        # Build final prompt for concluding misconception
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=final_system_prompt,
            state=state,
            include_last_message=True,
            include_instructions=False,
            parser=None,
            current_node="MH"
        )
        
        final_response = llm_with_history(state, final_prompt).content.strip()
        
        # 🔍 MH NODE - MAX TRIES REACHED 🔍
        print("=" * 80)
        print("🎯 MH NODE - MAX TRIES REACHED, LLM CONCLUSION 🎯")
        print("=" * 80)
        print(f"🔢 MH_TRIES: {state['mh_tries']}")
        print(f"💬 LLM_FINAL_MESSAGE: {final_response}")
        print("=" * 80)
        
        state["agent_output"] = final_response
        state["current_state"] = "SIM_VARS"  # After max tries, show simulation to help convince student
        add_ai_message_to_conversation(state, final_response)
        return state
    
    # Normal MH processing: evaluate student's response and decide next action
    context = json.dumps(PEDAGOGICAL_MOVES["MH"], indent=2)
    correction = state.get("last_correction", "the previous correction")
    
    system_prompt = f"""Current node: MH (Misconception Handling)
Possible next_state values:
- "MH": if the student still has doubts, questions, or shows continued misconception (max 2 tries total).
- "AR": if the student shows understanding and acceptance of the correction.

Pedagogical context:
{context}

Previous correction provided: {correction}

This is attempt {state["mh_tries"]} of 2 for misconception handling.

The student has received a correction for their misconception. Now they have responded. 
Analyze their response:
- If they seem to understand and accept the correction, move to AR for assessment
- If they still have questions, doubts, or show misconception, provide additional clarification and stay in MH
- Be encouraging and supportive while addressing their concerns

Task: Evaluate the student's response after receiving misconception correction. Respond ONLY with JSON matching the schema above."""

    # Build final prompt using optimized template with instructions
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=mh_parser,
        current_node="MH"
    )
    
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    try:
        parsed: MhResponse = mh_parser.parse(json_text)
        
        # 🔍 MH PARSING OUTPUT - MAIN CONTENT 🔍
        print("=" * 80)
        print("🎯 MH NODE - PARSED OUTPUT CONTENTS 🎯")
        print("=" * 80)
        print(f"📝 FEEDBACK: {parsed.feedback}")
        print(f"🚀 NEXT_STATE: {parsed.next_state}")
        print(f"🔢 MH_TRIES: {state['mh_tries']}")
        print(f"📊 PARSED_TYPE: {type(parsed).__name__}")
        print("=" * 80)
        
        state["agent_output"] = parsed.feedback
        # state["current_state"] = parsed.next_state
        state["current_state"] = "SIM_VARS"
        add_ai_message_to_conversation(state, parsed.feedback)
    except Exception as e:
        print(f"Error parsing MH response: {e}")
        print(f"Raw response: {raw}")
        print(f"Extracted JSON text: {json_text}")
        raise
    
    return state

def ar_node(state: AgentState) -> AgentState:
    # First pass: generate the quiz
    if not state.get("asked_ar", False):
        state["asked_ar"] = True
        current_idx = state.get("sim_current_idx", 0)
        concepts = state.get("sim_concepts", [])
        
        # Include ground truth for MCQs
        gt = get_ground_truth_from_json(state["concept_title"], "MCQs")
        
        if concepts and current_idx < len(concepts):
            current_concept = concepts[current_idx]
            system_prompt = f"""
                Please use the following ground truth as a baseline and build upon it, but do not deviate too much.
                Ground truth (MCQs):
            {gt}

                Generate a short quiz question (T/F, MCQ, or short answer) specifically about concept {current_idx + 1}: '{current_concept}' 
                within the topic '{state["concept_title"]}'. Focus the question on this specific concept.
            """
        else:
            system_prompt = f"""
                Please use the following ground truth as a baseline and build upon it, but do not deviate too much.
                Ground truth (MCQs):
{gt}
                Generate a short quiz question (T/F, MCQ, or short answer) on '{state["concept_title"]}' and prompt the learner."""
        
        # Build final prompt using optimized template
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=system_prompt,
            state=state,
            include_last_message=False,
            include_instructions=False,
            current_node="AR"
        )
        resp = llm_with_history(state, final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content

        # 🔍 AR NODE - FIRST PASS CONTENT 🔍
        print("=" * 80)
        print("🎯 AR NODE - FIRST PASS CONTENT OUTPUT 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔧 USED_JSON_EXTRACTION: {resp.content.strip().startswith('```')}")
        print(f"🔢 CURRENT_CONCEPT_IDX: {current_idx}")
        print("=" * 80)
        
        state["agent_output"] = content
        add_ai_message_to_conversation(state, content)
        return state

    # Second pass: grade & decide next step based on concept progress
    current_idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    
    context = json.dumps(PEDAGOGICAL_MOVES["AR"], indent=2)


    system_prompt = f"""Current node: AR (Application & Retrieval)

Current status:
- Concept {current_idx + 1} of {len(concepts) if concepts else 0}
- Concept name: {concepts[current_idx] if concepts and current_idx < len(concepts) else 'Unknown'}

Possible next_state values:
- "GE": if there are more concepts to explore 
- "TC": if all concepts have been covered and we should move to transfer

Pedagogical context:
{context}

Task: Grade this answer on a scale from 0 to 1 and determine next state. Respond ONLY with JSON matching the schema above."""

    # Build final prompt using optimized template with instructions at the end
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=ar_parser,
        current_node="AR",
        include_autosuggestions=True
    )

    
    raw = llm_with_history(state, final_prompt).content

    json_text = extract_json_block(raw)
    try:
        print("#############JSON TEXT HERE",json_text)
        parsed_obj: ArResponse = ar_parser.parse(json_text)
        parsed = parsed_obj.model_dump()
        if(current_idx==len(concepts)-1):
            print("#############LAST CONCEPT REACHED#############")
            parsed['next_state']="TC"
        else:
            print("#############MORE CONCEPTS TO GO#############")
            parsed['next_state']="GE"

        # 🔍 AR PARSING OUTPUT - MAIN CONTENT 🔍
        print("=" * 80)
        print("🎯 AR NODE - PARSED OUTPUT CONTENTS 🎯")
        print("=" * 80)
        print(f"📝 FEEDBACK: {parsed['feedback']}")
        print(f"📊 SCORE: {parsed['score']}")
        print(f"🚀 NEXT_STATE: {parsed['next_state']}")
        print(f"🎯 SCORE_TYPE: {type(parsed['score']).__name__}")
        print(f"📊 PARSED_TYPE: {type(parsed).__name__}")
        print(f"🔢 CURRENT_CONCEPT_IDX: {current_idx}")
        print("=" * 80)

        score, feedback, next_state = parsed['score'], parsed['feedback'], parsed['next_state']

        # Store the quiz score in the state for metrics
        state["quiz_score"] = score * 100  # Convert 0-1 score to 0-100 percentage
        
        # Combine all 4 autosuggestions (positive, negative, special, dynamic)
        final_suggestions, selections = combine_autosuggestions(
            parsed,
            ["I understand, continue", "I'm not sure", "Can you give me a hint?", "Continue exploring"]
        )
        
        state["positive_autosuggestion"] = selections['positive']
        state["negative_autosuggestion"] = selections['negative']
        state["special_handling_autosuggestion"] = selections['special']
        state["dynamic_autosuggestion"] = selections['dynamic']
        state["autosuggestions"] = final_suggestions
    except Exception as e:
        print(f"Error parsing AR response: {e}")
        print(f"Raw response: {raw}")
        print(f"Extracted JSON text: {json_text}")
        raise

    # Provide feedback based on score
    if score < 0.5:
        # Student struggled: give correct answer + explanation
        explain_system_prompt = (
            "Provide the correct answer to the quiz question and explain why it is correct in 2–3 sentences."
        )
        
        # Build final prompt using template
        explain_final_prompt = build_prompt_from_template_optimized(
            system_prompt=explain_system_prompt,
            state=state,
            include_last_message=True,
            include_instructions=False,
            current_node="AR"
        )
            
        resp = llm_with_history(state, explain_final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content

        # 🔍 AR NODE - EXPLANATION CONTENT 🔍
        print("=" * 80)
        print("🎯 AR NODE - EXPLANATION CONTENT OUTPUT 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔧 USED_JSON_EXTRACTION: {resp.content.strip().startswith('```')}")
        print("=" * 80)
        
        state["agent_output"] = content
    else:
        state["agent_output"] = feedback

    # Handle concept progression
    # next_state = "TC"
    if next_state == "GE":
        # Move to next concept
        state["sim_current_idx"] = current_idx + 1
        state["asked_ge"] = False  # Reset GE flag for next concept
        # Add transition message
        next_concept_idx = current_idx + 1
        if next_concept_idx < len(concepts):
            transition_msg = f"\n\nGreat! Now let's explore the next concept: '{concepts[next_concept_idx]}'."
            state["agent_output"] += transition_msg
    elif next_state == "TC":
        # All concepts done, move to transfer
        # state["concepts_completed"] = True
        completion_msg = "\n\nExcellent! We've covered all the key concepts. Now let's see how you can apply this knowledge in a new context."
        state["agent_output"] += completion_msg

    add_ai_message_to_conversation(state, state["agent_output"])
    state["current_state"] = next_state
    return state

def tc_node(state: AgentState) -> AgentState:
    # First pass: generate the transfer question
    if not state.get("asked_tc", False):
        # state["asked_tc"] = True
        # Include ground truth for What-if Scenarios
        gt = get_ground_truth_from_json(state["concept_title"], "What-if Scenarios")
        system_prompt = f"""
            Please use the following ground truth as a baseline and build upon it, but do not deviate too much.
            Ground truth (What-if Scenarios): {gt}
            Generate a 'what-if' or transfer question to apply '{state["concept_title"]}' in a new context.
        """
        
        # Build final prompt using template
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=system_prompt,
            state=state,
            include_last_message=False,
            include_instructions=False,
            current_node="TC"
        )
            
        resp = llm_with_history(state, final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content

        # 🔍 TC NODE - FIRST PASS CONTENT 🔍
        print("=" * 80)
        print("🎯 TC NODE - FIRST PASS CONTENT OUTPUT 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔧 USED_JSON_EXTRACTION: {resp.content.strip().startswith('```')}")
        print("=" * 80)
        
        state["agent_output"] = content
        state["asked_tc"] = True
        add_ai_message_to_conversation(state, content)
        return state

    # Second pass: evaluate & either affirm or explain
    context = json.dumps(PEDAGOGICAL_MOVES["TC"], indent=2)
    system_prompt = f"""Current node: TC (Transfer & Critical Thinking)
Possible next_state values (handled by agent code):
- "RLC": always move forward after feedback/explanation

Pedagogical context:
{context}

Task: Evaluate whether the application is correct. Respond ONLY with JSON matching the schema above."""

    # Build final prompt using optimized template with instructions at the end
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=tc_parser,
        current_node="TC",
        include_autosuggestions=True
    )
    
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    try:
        parsed_obj: TcResponse = tc_parser.parse(json_text)
        parsed = parsed_obj.model_dump()

        # 🔍 TC PARSING OUTPUT - MAIN CONTENT 🔍
        print("=" * 80)
        print("🎯 TC NODE - PARSED OUTPUT CONTENTS 🎯")
        print("=" * 80)
        print(f"📝 FEEDBACK: {parsed['feedback']}")
        print(f"✅ CORRECT: {parsed['correct']}")
        print(f"🎯 CORRECT_TYPE: {type(parsed['correct']).__name__}")
        print(f"📊 PARSED_TYPE: {type(parsed).__name__}")
        print("=" * 80)

        correct, feedback = parsed['correct'], parsed['feedback']
        
        # Combine all 4 autosuggestions (positive, negative, special, dynamic)
        final_suggestions, selections = combine_autosuggestions(
            parsed,
            ["I understand, continue", "I'm not sure", "Can you give me a hint?", "Continue exploring"]
        )
        
        # Store all 4 individual selections
        state["positive_autosuggestion"] = selections['positive']
        state["negative_autosuggestion"] = selections['negative']
        state["special_handling_autosuggestion"] = selections['special']
        state["dynamic_autosuggestion"] = selections['dynamic']
        state["autosuggestions"] = final_suggestions
    except Exception as e:
        print(f"Error parsing TC response: {e}")
        print(f"Raw response: {raw}")
        print(f"Extracted JSON text: {json_text}")
        raise

    if correct:
        state["agent_output"] = feedback + "\nExcellent application! You've mastered this concept."
        add_ai_message_to_conversation(state, state["agent_output"])
    else:
        # Student struggled: give correct transfer answer + explanation
        explain_system_prompt = (
            "Provide the correct answer to the transfer question, explain why it is correct in 2–3 sentences, "
            "and then say we are proceeding to see a real-life application."
        )
        
        # Build final prompt using template
        explain_final_prompt = build_prompt_from_template_optimized(
            system_prompt=explain_system_prompt,
            state=state,
            include_last_message=True,
            include_instructions=False,
            current_node="TC"
        )
            
        resp = llm_with_history(state, explain_final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content

        # 🔍 TC NODE - EXPLANATION CONTENT 🔍
        print("=" * 80)
        print("🎯 TC NODE - EXPLANATION CONTENT OUTPUT 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔧 USED_JSON_EXTRACTION: {resp.content.strip().startswith('```')}")
        print("=" * 80)
        
        state["agent_output"] = content
        add_ai_message_to_conversation(state, content)

    state["current_state"] = "RLC"
    return state

def rlc_node(state: AgentState) -> AgentState:
    # Ensure simulation flags are properly reset when entering RLC
    if state.get("show_simulation", False):
        state["show_simulation"] = False
        state["simulation_config"] = {}
    
    if not state.get("asked_rlc", False):
        state["asked_rlc"] = True
        state["rlc_tries"] = 0  # Initialize attempt counter
        # Include ground truth for Real-Life Application
        gt = get_ground_truth_from_json(state["concept_title"], "Real-Life Application")
        system_prompt = f"""
            Please use the following ground truth as a baseline and build upon it, but do not deviate too much.
            Ground truth (Real-Life Application): {gt}
            Provide a real-life application for '{state["concept_title"]}', then ask if the learner has seen or used it themselves.
        """
        
        # Build final prompt using template
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=system_prompt,
            state=state,
            include_last_message=False,
            include_instructions=False,
            current_node="RLC"
        )
            
        resp = llm_with_history(state, final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content

        # 🔍 RLC NODE - FIRST PASS CONTENT 🔍
        print("=" * 80)
        print("🎯 RLC NODE - FIRST PASS CONTENT OUTPUT 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔧 USED_JSON_EXTRACTION: {resp.content.strip().startswith('```')}")
        print("=" * 80)
        
        state["agent_output"] = content
        add_ai_message_to_conversation(state, content)
        return state

    # Increment attempt counter
    state["rlc_tries"] = state.get("rlc_tries", 0) + 1
    
    # Check if we've reached 2 attempts - if so, answer final doubts and move to END
    if state["rlc_tries"] >= 2:
        # Include ground truth for Real-Life Application to help answer any final questions
        # gt = get_ground_truth_from_json(state["concept_title"], "Real-Life Application")
        system_prompt = f"""
            The student has been discussing real-life applications of '{state["concept_title"]}' and this is their final interaction in this section. 
            Answer any remaining questions or doubts they might have about the real-life application thoroughly and helpfully. 
            After addressing their question/doubt, conclude by saying: 
            'Great! As a quick creative task, try drawing or explaining this idea to a friend and share what you notice. You've learned a lot today!'
        """
        
        # Build final prompt using template
        final_prompt = build_prompt_from_template_optimized(
            system_prompt=system_prompt,
            state=state,
            include_last_message=True,
            include_instructions=False,
            current_node="RLC"
        )
            
        resp = llm_with_history(state, final_prompt)
        # Apply JSON extraction in case LLM wraps response in markdown
        content = extract_json_block(resp.content) if resp.content.strip().startswith("```") else resp.content
        
        # 🔍 RLC NODE - FINAL ANSWER AND CONCLUSION 🔍
        print("=" * 80)
        print("🎯 RLC NODE - FINAL ANSWER AND CONCLUSION 🎯")
        print("=" * 80)
        print(f"📄 CONTENT: {content}")
        print(f"📏 CONTENT_LENGTH: {len(content)} characters")
        print(f"🔢 RLC_TRIES: {state['rlc_tries']}")
        print("=" * 80)
        
        state["agent_output"] = content
        state["current_state"] = "END"
        add_ai_message_to_conversation(state, content)
        return state

    context = json.dumps(PEDAGOGICAL_MOVES["RLC"], indent=2)
    system_prompt = f"""Current node: RLC (Real-Life Context)
Possible next_state values:
- "RLC": when the student is asking relevant questions about the real-life application and you want to continue the discussion.
- "END": when the student seems satisfied or has no more questions about the real-life application.

Pedagogical context:
{context}

This is attempt {state["rlc_tries"]} for the student in the RLC node. You can stay in this node for up to 2 attempts to answer questions about the real-life application before moving to END.

Task: Evaluate whether the student has more questions about the real-life application. If they're asking relevant questions, stay in RLC. If they seem satisfied or ready to move on, go to END. Respond ONLY with JSON matching the schema above."""

    # Build final prompt using optimized template with instructions at the end
    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=rlc_parser,
        current_node="RLC",
        include_autosuggestions=True
    )
    
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    try:
        parsed_obj: RlcResponse = rlc_parser.parse(json_text)
        parsed = parsed_obj.model_dump()

        # 🔍 RLC PARSING OUTPUT - MAIN CONTENT 🔍
        print("=" * 80)
        print("🎯 RLC NODE - PARSED OUTPUT CONTENTS 🎯")
        print("=" * 80)
        print(f"📝 FEEDBACK: {parsed['feedback']}")
        print(f"🚀 NEXT_STATE: {parsed['next_state']}")
        print(f"🔢 RLC_TRIES: {state['rlc_tries']}")
        print(f"📊 PARSED_TYPE: {type(parsed).__name__}")
        print("=" * 80)

        # Combine all 4 autosuggestions (positive, negative, special, dynamic)
        final_suggestions, selections = combine_autosuggestions(
            parsed,
            ["I understand, continue", "I'm not sure", "Can you give me a hint?", "Continue exploring"]
        )
        
        state["agent_output"]  = parsed['feedback']
        state["current_state"] = parsed['next_state']
        state["positive_autosuggestion"] = selections['positive']
        state["negative_autosuggestion"] = selections['negative']
        state["special_handling_autosuggestion"] = selections['special']
        state["dynamic_autosuggestion"] = selections['dynamic']
        state["autosuggestions"] = final_suggestions
        add_ai_message_to_conversation(state, parsed['feedback'])
    except Exception as e:
        print(f"Error parsing RLC response: {e}")
        print(f"Raw response: {raw}")
        print(f"Extracted JSON text: {json_text}")
        raise
    return state

def end_node(state: AgentState) -> AgentState:
    # build debug-rich summary
    state["session_summary"] = {
        "quiz_score":             state.get("retrieval_score"),
        "transfer_success":       state.get("transfer_success"),
        "definition_echoed":      state.get("definition_echoed"),
        "misconception_detected": state.get("misconception_detected"),
        "last_user_msg":          state.get("last_user_msg"),
        "history":                state.get("history"),
        "status":                 "completed"  # Mark summary as final
    }

    # 🔍 END NODE - SESSION SUMMARY 🔍
    print("=" * 80)
    print("🎯 END NODE - SESSION SUMMARY CONTENTS 🎯")
    print("=" * 80)
    print(f"📊 QUIZ_SCORE: {state['session_summary']['quiz_score']}")
    print(f"🎯 TRANSFER_SUCCESS: {state['session_summary']['transfer_success']}")
    print(f"📝 DEFINITION_ECHOED: {state['session_summary']['definition_echoed']}")
    print(f"🔍 MISCONCEPTION_DETECTED: {state['session_summary']['misconception_detected']}")
    print(f"💬 LAST_USER_MSG: {state['session_summary']['last_user_msg']}")
    print(f"📚 HISTORY_LENGTH: {len(str(state['session_summary']['history'])) if state['session_summary']['history'] else 0} characters")
    print("=" * 80)

    # final output
    state["agent_output"] = (
        "Great work today! Here's your session summary:\n"
        f"- Quiz score: {state['session_summary']['quiz_score']}\n"
        f"- Transfer success: {state['session_summary']['transfer_success']}\n"
        f"- Definition echoed: {state['session_summary']['definition_echoed']}\n"
        f"- Misconception detected: {state['session_summary']['misconception_detected']}\n"
        f"- Last user message: {state['session_summary']['last_user_msg']}"
    )
    return state

