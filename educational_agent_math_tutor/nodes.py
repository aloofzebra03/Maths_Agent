"""
Node implementations for the Math Tutoring Agent.

Contains all pedagogical nodes: START, ASSESSMENT, ADAPTIVE_SOLVER, REFLECTION.
"""

import json
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser

from educational_agent_math_tutor.schemas import (
    MathAgentState,
    AssessmentResponse,
    CoachResponse,
    GuidedResponse,
    ScaffoldResponse,
    ScaffoldRevealResponse,
    ConceptResponse,
    ReflectionResponse,
    ConceptCheckResponse,
    ApproachAssessmentResponse,
    ConceptEvaluationResponse,
    StartAnswerCheckResponse,
    HandleSummaryResponse,
)
from educational_agent_math_tutor.prompts import (
    START_SYSTEM_PROMPT,
    START_GREETING_TEMPLATE,
    ASSESSMENT_SYSTEM_PROMPT,
    ASSESSMENT_USER_TEMPLATE,
    COACH_SYSTEM_PROMPT,
    COACH_USER_TEMPLATE,
    GUIDED_SYSTEM_PROMPT,
    GUIDED_USER_TEMPLATE,
    SCAFFOLD_SYSTEM_PROMPT,
    SCAFFOLD_USER_TEMPLATE,
    SCAFFOLD_REVEAL_SYSTEM_PROMPT,
    SCAFFOLD_REVEAL_USER_TEMPLATE,
    CONCEPT_SYSTEM_PROMPT,
    CONCEPT_USER_TEMPLATE,
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_USER_TEMPLATE,
    CONCEPT_CHECK_SYSTEM_PROMPT,
    CONCEPT_CHECK_USER_TEMPLATE,
    RE_ASK_SYSTEM_PROMPT,
    RE_ASK_USER_TEMPLATE,
    APPROACH_ASSESSMENT_SYSTEM_PROMPT,
    APPROACH_ASSESSMENT_USER_TEMPLATE,
    CONCEPT_EVALUATE_SYSTEM_PROMPT_EARLY,
    CONCEPT_EVALUATE_USER_TEMPLATE_EARLY,
    CONCEPT_EVALUATE_SYSTEM_PROMPT_FINAL,
    CONCEPT_EVALUATE_USER_TEMPLATE_FINAL,
    START_ANSWER_CHECK_SYSTEM_PROMPT,
    START_ANSWER_CHECK_USER_TEMPLATE,
    HANDLE_STEP_EXPLANATION_SYSTEM_PROMPT,
    HANDLE_STEP_EXPLANATION_USER_TEMPLATE,
    HANDLE_SUMMARY_REQUEST_SYSTEM_PROMPT,
    HANDLE_SUMMARY_REQUEST_USER_TEMPLATE,
)
from educational_agent_math_tutor.config import (
    TA_THRESHOLD_HIGH,
    TU_THRESHOLD_HIGH,
    MAX_COACH_NUDGES,
    MAX_SCAFFOLD_RETRIES,
    MAX_CONCEPT_VISITS_PER_CONCEPT,
    MAX_CONCEPT_INTERACTIONS,
)
from utils.shared_utils import (
    invoke_llm_with_fallback,
    extract_json_block,
    load_problem_from_json,
    format_required_concepts,
    build_messages_with_history,
    translate_if_kannada,
)


def _repair_scaffold_json_response(
    state: MathAgentState,
    raw_response: str,
    format_instructions: str,
) -> ScaffoldResponse:
    """
    Attempt one strict JSON repair pass for SCAFFOLD output.

    The original scaffold call may return plain conversational text. This helper
    asks the LLM to convert that content into the exact ScaffoldResponse JSON.
    If repair parsing fails, the caller should raise.
    """
    parser = PydanticOutputParser(pydantic_object=ScaffoldResponse)
    repair_system_prompt = (
        "You convert tutor text into strict JSON. "
        "Output ONLY one valid JSON object. No markdown fences. No extra text."
    )
    repair_user_prompt = (
        "Convert the following tutor output to ScaffoldResponse JSON. "
        "Preserve pedagogical intent and tone in response_to_student. "
        "If output is evaluation, set is_correct and should_advance consistently.\n\n"
        f"Tutor output:\n{raw_response}"
    )

    repair_messages = build_messages_with_history(
        state=state,
        system_prompt=repair_system_prompt,
        user_prompt=repair_user_prompt,
        format_instructions=format_instructions,
        include_last_message=False,
    )
    repair_response = invoke_llm_with_fallback(repair_messages, "SCAFFOLD_JSON_REPAIR")
    repair_json_str = extract_json_block(repair_response.content)
    return parser.parse(repair_json_str)


# ============================================================================
# START NODE
# ============================================================================

def start_node(state: MathAgentState) -> Dict[str, Any]:
    """
    START node: Load problem from JSON and present it to the student.
    
    Asks student for:
    1. What they understand from the question
    2. What approach they would use
    
    Returns:
        Partial state update with problem data and greeting message
    """
    print("\n" + "="*60)
    print("🚀 START NODE")
    print("="*60)
    
    # state['problem_id'] = 'add_frac_diff_den_01'
    problem_id = state.get("problem_id",'add_frac_diff_den_01')
    # print(f"📚 Loading problem ID: {problem_id}")
    if not problem_id:
        raise ValueError("No problem_id provided in initial state")
    
    # Load problem from JSON
    try:
        problem_data = load_problem_from_json(problem_id)
    except Exception as e:
        print(f"❌ Error loading problem: {e}")
        raise
    
    # Extract problem components
    question = problem_data["question"]
    steps = problem_data["canonical_solution"]["steps"]
    required_concepts = problem_data.get("required_concepts", [])
    final_answer = problem_data.get("final_answer", "")
    
    # Create greeting message
    greeting = START_GREETING_TEMPLATE.format(problem=question)
    translated_greeting = translate_if_kannada(state, greeting)
    
    # Store only the AI greeting in messages.
    # Do NOT store a SystemMessage in state — each node adds its own fresh system
    # prompt via build_messages_with_history. Storing a START SystemMessage here
    # would cause every downstream LLM call to receive TWO conflicting system prompts.
    messages = [
        AIMessage(content=translated_greeting)
    ]
    
    print(f"✅ Loaded problem: {problem_id}")
    print(f"📝 Question: {question}")
    print(f"📊 Steps: {len(steps)}")
    print(f"🎯 Required concepts: {format_required_concepts(required_concepts)}")
    
    return {
        "agent_output": translated_greeting,
        "problem": question,
        "problem_id": problem_id,
        "steps": steps,
        "max_steps": len(steps),
        "step_index": 0,
        "solved": False,
        "Ta": 0.0,
        "Tu": 0.0,
        "nudge_count": 0,
        "scaffold_retry_count": 0,
        "node_transitions": [],
        "messages": messages,
        "current_state": "START",
        # New concept tracking fields
        "missing_concepts": [],
        "concepts_taught": [],
        "concept_visit_count": {},
        "concept_interaction_count": 0,
        "post_concept_reassessment": False,
        "asked_concept": False,
        "concept_tries": 0,
        # Start-node answer check fields
        "start_attempt_count": 0,
        "awaiting_step_explanation": False,
        "final_answer": final_answer,
    }


# ============================================================================
# CHECK ANSWER NODE (quick-check right after START)
# ============================================================================

def check_answer_node(state: MathAgentState) -> Dict[str, Any]:
    """
    CHECK_ANSWER node: Evaluate student's first attempt at the problem.

    - Attempt 1 wrong: say 'That is incorrect', give a hint, stay in CHECK_ANSWER
    - Attempt 2 wrong: say 'That is incorrect', route to normal ASSESSMENT flow
    - Correct (any attempt): congratulate and ask about step explanation

    The correct answer is passed to the LLM in context ONLY — never revealed.
    """
    print("\n" + "="*60)
    print("✅ CHECK ANSWER NODE")
    print("="*60)

    messages = state.get("messages", [])
    problem = state["problem"]
    final_answer = state.get("final_answer", "")
    attempt = state.get("start_attempt_count", 0) + 1  # increment for this attempt

    # Get student's latest response
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    if not user_input:
        print("⚠️ No user input found")
        return {"current_state": "CHECK_ANSWER"}

    print(f"📝 Attempt {attempt}/2 — evaluating student answer")

    # Build prompt
    parser = PydanticOutputParser(pydantic_object=StartAnswerCheckResponse)
    format_instructions = parser.get_format_instructions()

    user_msg = START_ANSWER_CHECK_USER_TEMPLATE.format(
        problem=problem,
        final_answer=final_answer,
        student_response=user_input,
        attempt_number=attempt,
    )

    check_messages = build_messages_with_history(
        state=state,
        system_prompt=START_ANSWER_CHECK_SYSTEM_PROMPT,
        user_prompt=user_msg,
        format_instructions=format_instructions,
    )

    response = invoke_llm_with_fallback(check_messages, "CHECK_ANSWER")

    # Parse response
    try:
        json_str = extract_json_block(response.content)
        check_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Parse error: {e}")
        print(f"Raw response: {response.content}")
        raise RuntimeError("Failed to parse CHECK_ANSWER response") from e

    feedback = translate_if_kannada(state, check_resp.feedback)
    messages.append(AIMessage(content=feedback))

    if check_resp.is_correct:
        print("✅ Correct! Asking about step explanation.")
        return {
            "start_attempt_count": attempt,
            "awaiting_step_explanation": True,
            "agent_output": feedback,
            "messages": messages,
            "current_state": "CHECK_ANSWER",
        }
    elif attempt >= 2:
        print("❌ Wrong on attempt 2 — proceeding to normal flow.")
        return {
            "start_attempt_count": attempt,
            "awaiting_step_explanation": False,
            "agent_output": feedback,
            "messages": messages,
            "current_state": "CHECK_ANSWER",
        }
    else:
        print(f"❌ Wrong on attempt {attempt} — staying for retry.")
        return {
            "start_attempt_count": attempt,
            "awaiting_step_explanation": False,
            "agent_output": feedback,
            "messages": messages,
            "current_state": "CHECK_ANSWER",
        }


# ============================================================================
# HANDLE STEP EXPLANATION NODE
# ============================================================================

def handle_step_explanation_node(state: MathAgentState) -> Dict[str, Any]:
    """
    HANDLE_STEP_EXPLANATION node: Student answered correctly at start.
    Reads their yes/no to step explanation and routes accordingly.

    - Yes → proceed to ASSESSMENT (normal tutoring flow)
    - No  → proceed to REFLECTION → END
    """
    print("\n" + "="*60)
    print("📖 HANDLE STEP EXPLANATION NODE")
    print("="*60)

    messages = state.get("messages", [])

    # Get student's reply
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    if not user_input:
        print("⚠️ No user input found")
        return {"current_state": "HANDLE_STEP_EXPLANATION"}

    user_msg = HANDLE_STEP_EXPLANATION_USER_TEMPLATE.format(student_reply=user_input)

    step_messages = build_messages_with_history(
        state=state,
        system_prompt=HANDLE_STEP_EXPLANATION_SYSTEM_PROMPT,
        user_prompt=user_msg,
    )

    response = invoke_llm_with_fallback(step_messages, "HANDLE_STEP_EXPLANATION")

    # Parse the small JSON manually
    wants_explanation = False
    reply_text = ""
    try:
        json_str = extract_json_block(response.content)
        parsed = json.loads(json_str)
        wants_explanation = bool(parsed.get("wants_explanation", False))
        reply_text = parsed.get("response", "")
    except Exception as e:
        print(f"❌ Parse error: {e}")
        print(f"Raw response: {response.content}")
        raise RuntimeError("Failed to parse HANDLE_STEP_EXPLANATION response") from e

    translated_reply = translate_if_kannada(state, reply_text)
    messages.append(AIMessage(content=translated_reply))

    print(f"📊 wants_explanation={wants_explanation}")

    if wants_explanation:
        # Ask the student to describe their approach so ASSESS_APPROACH has input to evaluate
        follow_up = translate_if_kannada(
            state,
            "Before I walk you through it — what do you think the steps involved are? Give it your best shot!"
        )
        messages.append(AIMessage(content=follow_up))
        agent_out = follow_up
    else:
        agent_out = translated_reply

    return {
        "awaiting_step_explanation": False,
        "wants_step_explanation": wants_explanation,
        "agent_output": agent_out,
        "messages": messages,
        "current_state": "HANDLE_STEP_EXPLANATION",
    }



# ============================================================================
# ASSESSMENT NODE
# ============================================================================


def assess_student_response(state: MathAgentState) -> Dict[str, Any]:
    """
    ASSESSMENT node: Check if student knows required prerequisite concepts.
    
    This is the initial assessment that determines if we need to teach
    concepts before proceeding to solve the problem.
    
    Routes to:
    - CONCEPT node if missing concepts detected
    - ASSESS_APPROACH node if all concepts understood
    
    Returns:
        Partial state update with missing_concepts list
    """
    print("\n" + "="*60)
    print("📊 ASSESSMENT NODE - Concept Check")
    print("="*60)
    
    # Get student's response from last message
    messages = state.get("messages", [])
    if not messages:
        print("⚠️ No messages in state, skipping assessment")
        return {}
    
    # Find last HumanMessage
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    
    if not user_input:
        print("⚠️ No user input found, skipping assessment")
        return {}
    
    problem = state["problem"]
    
    # Load problem data to get required concepts
    problem_id = state.get("problem_id")
    print(f"📚 Loading problem data for ID: {problem_id}")
    problem_data = load_problem_from_json(problem_id)
    required_concepts = format_required_concepts(problem_data.get("required_concepts", []))
    
    # Build concept check prompt
    parser = PydanticOutputParser(pydantic_object=ConceptCheckResponse)
    format_instructions = parser.get_format_instructions()

    concept_check_user_msg = CONCEPT_CHECK_USER_TEMPLATE.format(
        problem=problem,
        required_concepts=required_concepts,
        user_input=user_input
    )

    # Build messages with conversation history
    # remove_problem_messages=False: keep the full dialogue visible so the LLM
    # can accurately judge what the student said about the problem.
    concept_check_messages = build_messages_with_history(
        state=state,
        system_prompt=CONCEPT_CHECK_SYSTEM_PROMPT,
        user_prompt=concept_check_user_msg,
        format_instructions=format_instructions,
    )
    
    # Invoke LLM
    print("🤖 Calling LLM for concept check...")
    response = invoke_llm_with_fallback(concept_check_messages, "CONCEPT_CHECK")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        concept_check = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing concept check response: {e}")
        print(f"Raw response: {response.content}")
        raise RuntimeError("Failed to parse CONCEPT_CHECK response") from e
    
    print(f"📊 Concept Check Results:")
    print(f"   Missing Concepts: {concept_check.missing_concepts or 'None'}")
    print(f"   Reasoning: {concept_check.reasoning}")
    
    if concept_check.missing_concepts:
        print(f"🎯 Routing to CONCEPT node (missing: {', '.join(concept_check.missing_concepts)})")
    else:
        print(f"✅ All concepts understood, routing to ASSESS_APPROACH")
    
    translated_message = translate_if_kannada(state, concept_check.response_to_student)
    messages.append(AIMessage(content=translated_message))
    
    return {
        "missing_concepts": concept_check.missing_concepts,
        "last_user_msg": user_input,
        "agent_output": translated_message,
        "messages": messages,
        "current_state": "ASSESSMENT",
    }


# ============================================================================
# CONCEPT NODE (Standalone - Not a Mode)
# ============================================================================

def concept_node(state: MathAgentState) -> Dict[str, Any]:
    """
    CONCEPT node: Teach missing prerequisite concepts with interactive micro-checks.
    
    Uses try-counter pattern (max 3 tries per concept):
    - First call: Teach concept + ask micro-check
    - Subsequent calls: Evaluate response + decide (stay or move on) in SINGLE LLM call
    
    Features:
    - Single LLM call evaluation (efficient)
    - Different prompts based on try count
    - Forced transition at try 3
    - Tracks visit count per concept (max 2 visits)
    
    Returns:
        Partial state update with concepts_taught and updated counters
    """
    print("\n" + "="*60)
    print("💡 CONCEPT NODE")
    print("="*60)
    
    messages = state.get("messages", [])
    # We intentionally do NOT expose the problem to concept teaching logic,
    # but we keep a reference here for the forced-final-try prompt only.
    problem = state.get("problem", "")
    missing_concepts = state.get("missing_concepts", [])
    concepts_taught = state.get("concepts_taught", [])
    concept_visit_count = state.get("concept_visit_count", {})
    
    if not missing_concepts:
        print("⚠️ No missing concepts specified, should not be in concept node")
        return {
            "current_state": "CONCEPT",
        }
    
    # Get the first concept to teach (we teach one at a time)
    current_concept = missing_concepts[0]
    
    # ============================================
    # FIRST TIME: Teach concept + ask micro-check
    # ============================================
    if not state.get("asked_concept", False):
        print(f"📚 First time teaching concept: {current_concept}")
        
        # Update visit count for this concept
        visits = concept_visit_count.get(current_concept, 0)
        concept_visit_count[current_concept] = visits + 1
        print(f"📊 Visit count: {concept_visit_count[current_concept]}/{MAX_CONCEPT_VISITS_PER_CONCEPT}")
        
        # Build concept teaching prompt
        parser = PydanticOutputParser(pydantic_object=ConceptResponse)
        format_instructions = parser.get_format_instructions()
        
        concept_user_msg = CONCEPT_USER_TEMPLATE.format(
            missing_concept=current_concept,
        )
        
        # Build messages with conversation history
        concept_messages = build_messages_with_history(
            state=state,
            system_prompt=CONCEPT_SYSTEM_PROMPT,
            user_prompt=concept_user_msg,
            format_instructions=format_instructions,
            remove_problem_messages=True
        )
        
        # Invoke LLM
        print(f"🤖 Calling LLM to teach concept: {current_concept}")
        response = invoke_llm_with_fallback(concept_messages, "CONCEPT")
        
        # Parse response
        try:
            json_str = extract_json_block(response.content)
            concept_resp = parser.parse(json_str)
        except Exception as e:
            print(f"❌ Error parsing concept response: {e}")
            print(f"Raw response: {response.content}")
            raise RuntimeError("Failed to parse CONCEPT response") from e
        
        # Use the natural teaching response directly
        response_message = concept_resp.teaching_response
        translated_message = translate_if_kannada(state, response_message)
        
        messages.append(AIMessage(content=translated_message))
        
        print("✅ Concept taught, waiting for student response")
        
        return {
            "asked_concept": True,
            "concept_tries": 0,
            "concept_visit_count": concept_visit_count,
            "agent_output": translated_message,
            "messages": messages,
            "current_state": "CONCEPT",
        }
    
    # ============================================
    # SUBSEQUENT TIMES: Evaluate + Respond (SINGLE LLM CALL)
    # ============================================

    # Get user's latest response
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    if not user_input:
        print("⚠️ No user input found, waiting for student response")
        return {
            "current_state": "CONCEPT",
        }

    # Extract the micro-check question: the last AIMessage the tutor sent.
    # Passing this explicitly prevents the evaluator LLM from having to search
    # through the full conversation history to find what question was asked —
    # which was causing it to miss the question and incorrectly mark answers
    # as correct (e.g. student answered -1 to "3 × -5 = ?" but LLM said ✅).
    micro_check_question = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            micro_check_question = msg.content
            break
    print(f"🔍 Micro-check question extracted: {micro_check_question[:80]}...")

    # Increment tries
    tries = state.get("concept_tries", 0) + 1
    print(f"📊 Evaluating student response (try {tries}/3)")

    # Build evaluation prompt based on try count
    parser = PydanticOutputParser(pydantic_object=ConceptEvaluationResponse)
    format_instructions = parser.get_format_instructions()

    if tries < 3:
        # Early tries: can re-teach if needed
        system_prompt = CONCEPT_EVALUATE_SYSTEM_PROMPT_EARLY.format(tries=tries)
        user_msg = CONCEPT_EVALUATE_USER_TEMPLATE_EARLY.format(
            concept=current_concept,
            micro_check_question=micro_check_question,
            student_response=user_input,
        )
    else:
        # Final try: must move on
        system_prompt = CONCEPT_EVALUATE_SYSTEM_PROMPT_FINAL
        user_msg = CONCEPT_EVALUATE_USER_TEMPLATE_FINAL.format(
            concept=current_concept,
            micro_check_question=micro_check_question,
            problem=problem,
            student_response=user_input,
        )

    # Build messages with conversation history
    eval_messages = build_messages_with_history(
        state=state,
        system_prompt=system_prompt,
        user_prompt=user_msg,
        format_instructions=format_instructions,
        remove_problem_messages=True
    )
    
    # Invoke LLM for evaluation
    print(f"🤖 Calling LLM to evaluate understanding...")
    response = invoke_llm_with_fallback(eval_messages, "CONCEPT_EVAL")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        eval_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing evaluation response: {e}")
        print(f"Raw response: {response.content}")
        raise RuntimeError("Failed to parse CONCEPT_EVAL response") from e
    
    print(f"📊 Evaluation: understood={eval_resp.understood}, next_state={eval_resp.next_state}")
    
    # Add LLM's response to conversation

    ai_message = eval_resp.response_to_student

    # Decide next action based on evaluation
    if eval_resp.next_state == "move_on":
        # Student understood OR max tries reached
        print(f"✅ Moving on from concept: {current_concept}")

        # Add to concepts taught if not already there
        if current_concept not in concepts_taught:
            concepts_taught.append(current_concept)

        # Remove from missing concepts
        remaining_concepts = missing_concepts[1:]

        # Reset flags for next concept
        if remaining_concepts:
            # Build transition message entirely in English first, then translate.
            # Do NOT concatenate English strings directly into ai_message if ai_message
            # is already translated — translate each piece independently.
            next_concept_name = remaining_concepts[0].replace('_', ' ').title()
            transition_suffix = f" Now let's look at {next_concept_name}."
            translated_message = translate_if_kannada(state, ai_message + transition_suffix)
            messages.append(AIMessage(content=translated_message))
            print(f"📚 Next concept: {remaining_concepts[0]}")
            return {
                "missing_concepts": remaining_concepts,
                "concepts_taught": concepts_taught,
                "asked_concept": False,  # Reset for next concept
                "concept_tries": 0,
                "agent_output": translated_message,
                "messages": messages,
                "current_state": "CONCEPT",
            }
        else:
            # All concepts done — translate the whole final message as one unit
            print("✅ All concepts taught!")
            done_suffix = " Great work — you've got the building blocks. Let me now ask you about the problem again."
            translated_message = translate_if_kannada(state, ai_message + done_suffix)
            messages.append(AIMessage(content=translated_message))
            return {
                "missing_concepts": [],
                "concepts_taught": concepts_taught,
                "asked_concept": False,
                "concept_tries": 0,
                "agent_output": translated_message,
                "messages": messages,
                "current_state": "CONCEPT",
            }
    
    else:  # next_state == "stay"
        # Need to re-teach - stay in CONCEPT node
        print(f"🔄 Re-teaching concept: {current_concept} (try {tries}/3)")
        translated_message = translate_if_kannada(state, ai_message)
        messages.append(AIMessage(content=translated_message))
        return {
            "concept_tries": tries,
            "agent_output": translated_message,
            "messages": messages,
            "current_state": "CONCEPT",
        }


# ============================================================================
# RE-ASK NODE
# ============================================================================

def re_ask_start_questions_node(state: MathAgentState) -> Dict[str, Any]:
    """
    RE_ASK node: After teaching concepts, re-ask the same START questions.
    
    Asks the student the same questions from START node:
    1. What do you understand from this question?
    2. What approach would you use?
    
    Returns:
        Partial state update with re-ask message
    """
    print("\n" + "="*60)
    print("🔄 RE-ASK NODE")
    print("="*60)
    
    messages = state.get("messages", [])
    problem = state["problem"]
    concepts_taught = state.get("concepts_taught", [])
    
    # Build re-ask prompt using build_messages_with_history so the LLM has
    # the full concept-teaching dialogue as context.
    re_ask_user_msg = RE_ASK_USER_TEMPLATE.format(
        problem=problem,
        concepts_taught=", ".join(concepts_taught) if concepts_taught else "some key concepts"
    )

    re_ask_messages = build_messages_with_history(
        state=state,
        system_prompt=RE_ASK_SYSTEM_PROMPT,
        user_prompt=re_ask_user_msg,
        remove_problem_messages=True  # Strip stale START SystemMessage from history
    )

    # Invoke LLM
    print(f"🤖 Calling LLM to re-ask questions after teaching: {concepts_taught}")
    response = invoke_llm_with_fallback(re_ask_messages, "RE_ASK")

    response_message = response.content
    translated_message = translate_if_kannada(state, response_message)
    messages.append(AIMessage(content=translated_message))

    print("✅ Re-asked START questions")

    return {
        "agent_output": translated_message,
        "messages": messages,
        "post_concept_reassessment": True,  # Flag that we've re-asked
        "current_state": "RE_ASK",
    }


# ============================================================================
# ASSESS APPROACH NODE
# ============================================================================

def assess_approach_node(state: MathAgentState) -> Dict[str, Any]:
    """
    ASSESS_APPROACH node: Score Tu/Ta and route to appropriate pedagogical mode.
    
    This runs after:
    - Concept teaching (via RE_ASK)
    - Initial assessment (if no concepts missing)
    - During solving loop (to check progress)
    
    Routes to mode based on scores:
    - Ta ≥ 0.6 AND Tu ≥ 0.6 → "coach"
    - Ta < 0.6 AND Tu < 0.6 → "scaffold"
    - else → "guided"
    
    Returns:
        Partial state update with Ta, Tu, mode
    """
    print("\n" + "="*60)
    print("📊 ASSESS APPROACH NODE")
    print("="*60)
    
    # Get student's response from last message
    messages = state.get("messages", [])
    if not messages:
        print("⚠️ No messages in state, skipping assessment")
        return {}
    
    # Find last HumanMessage
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    
    if not user_input:
        print("⚠️ No user input found, skipping assessment")
        return {}
    
    problem = state["problem"]
    concepts_taught = state.get("concepts_taught", [])
    
    # Build approach assessment prompt
    parser = PydanticOutputParser(pydantic_object=ApproachAssessmentResponse)
    format_instructions = parser.get_format_instructions()
    
    context = f"Student has learned: {', '.join(concepts_taught)}" if concepts_taught else "Initial assessment"
    
    approach_user_msg = APPROACH_ASSESSMENT_USER_TEMPLATE.format(
        problem=problem,
        user_input=user_input,
        context=context
    )
    
    # Build messages with conversation history
    approach_messages = build_messages_with_history(
        state=state,
        system_prompt=APPROACH_ASSESSMENT_SYSTEM_PROMPT,
        user_prompt=approach_user_msg,
        format_instructions=format_instructions
    )
    
    # Invoke LLM
    print("🤖 Calling LLM for approach assessment...")
    response = invoke_llm_with_fallback(approach_messages, "APPROACH_ASSESSMENT")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        assessment = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing approach assessment response: {e}")
        print(f"Raw response: {response.content}")
        raise RuntimeError("Failed to parse APPROACH_ASSESSMENT response") from e
    
    print(f"📊 Approach Assessment Results:")
    print(f"   Tu (Understanding): {assessment.Tu:.2f}")
    print(f"   Ta (Approach): {assessment.Ta:.2f}")
    print(f"   Reasoning: {assessment.reasoning}")
    
    # Determine mode based on assessment
    if assessment.Ta >= TA_THRESHOLD_HIGH and assessment.Tu >= TU_THRESHOLD_HIGH:
        mode = "coach"
        print(f"🎯 Routing to COACH mode (strong understanding & approach)")
    elif assessment.Ta < TA_THRESHOLD_HIGH and assessment.Tu < TU_THRESHOLD_HIGH:
        mode = "scaffold"
        print(f"🎯 Routing to SCAFFOLD mode (needs step-by-step guidance)")
    else:
        mode = "guided"
        print(f"🎯 Routing to GUIDED mode (partial understanding)")
    
    return {
        "Tu": assessment.Tu,
        "Ta": assessment.Ta,
        "mode": mode,
        "last_user_msg": user_input,
        "current_state": "ASSESS_APPROACH",
    }


# ============================================================================
# ADAPTIVE SOLVER NODE
# ============================================================================

def adaptive_solver(state: MathAgentState) -> Dict[str, Any]:
    """
    ADAPTIVE_SOLVER node: Route to mode-specific pedagogy.
    
    Internal dispatcher that calls:
    - _coach_logic() for coach mode
    - _guided_logic() for guided mode
    - _scaffold_logic() for scaffold mode
    
    Note: concept teaching is now handled by standalone concept_node,
    not as a mode within adaptive_solver.
    
    Returns:
        Partial state update from the specific mode logic
    """
    print("\n" + "="*60)
    print("🎓 ADAPTIVE SOLVER NODE")
    print("="*60)
    
    mode = state.get("mode", "guided")
    print(f"🎯 Current mode: {mode}")
    
    if mode == "coach":
        return _coach_logic(state)
    elif mode == "guided":
        return _guided_logic(state)
    elif mode == "scaffold":
        return _scaffold_logic(state)
    else:
        print(f"⚠️ Unknown mode: {mode}, defaulting to guided")
        return _guided_logic(state)


def _coach_logic(state: MathAgentState) -> Dict[str, Any]:
    """
    COACH mode: Validate student work and ask reflective questions if wrong.
    
    - Praise effort and thinking
    - If wrong: ask "why" questions (max 3 nudges)
    - If still wrong after 3 nudges: downgrade to guided mode
    - If correct: mark as solved
    """
    print("\n🏆 COACH MODE")

    
    messages = state.get("messages", [])
    problem = state["problem"]
    nudge_count = state.get("nudge_count", 0)
    steps = state.get("steps", [])
    
    # Get user's latest response
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    
    # Build coach prompt with conversation history
    parser = PydanticOutputParser(pydantic_object=CoachResponse)
    format_instructions = parser.get_format_instructions()
    
    step_context = "Full problem solving" if not steps else f"Working through {len(steps)} steps"
    
    coach_user_msg = COACH_USER_TEMPLATE.format(
        problem=problem,
        step_context=step_context,
        user_input=user_input,
        nudge_count=nudge_count
    )
    
    # Build messages with conversation history
    coach_messages = build_messages_with_history(
        state=state,
        system_prompt=COACH_SYSTEM_PROMPT,
        user_prompt=coach_user_msg,
        format_instructions=format_instructions
    )
    
    # Invoke LLM
    print("🤖 Calling LLM for coach response...")
    response = invoke_llm_with_fallback(coach_messages, "COACH")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        coach_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing coach response: {e}")
        print(f"Raw response: {response.content}")
        raise RuntimeError("Failed to parse COACH response") from e
    
    # Build response message
    response_parts = [coach_resp.validation]
    
    update_dict = {
        "current_state": "ADAPTIVE_SOLVER",
    }
    
    if coach_resp.is_correct:
        print("✅ Student answer is correct!")
        response_parts.append(coach_resp.encouragement)
        response_message = translate_if_kannada(state, "\n\n".join(response_parts))
        
        update_dict["solved"] = True
        update_dict["agent_output"] = response_message
        messages.append(AIMessage(content=response_message))
        update_dict["messages"] = messages
        
    else:
        print(f"❌ Student answer incorrect (nudge {nudge_count + 1}/{MAX_COACH_NUDGES})")
        
        if nudge_count >= MAX_COACH_NUDGES:
            print("⬇️ Max nudges reached, downgrading to GUIDED mode")
            response_parts.append("I can see you're working hard on this. Let me give you some more specific help.")
            response_message = translate_if_kannada(state, "\n\n".join(response_parts))
            
            update_dict["mode"] = "guided"
            update_dict["nudge_count"] = 0  # Reset for potential future use
            update_dict["agent_output"] = response_message
            messages.append(AIMessage(content=response_message))
            update_dict["messages"] = messages
            
        else:
            # Ask reflective question
            if coach_resp.reflective_question:
                response_parts.append(coach_resp.reflective_question)
            response_parts.append(coach_resp.encouragement)
            response_message = translate_if_kannada(state, "\n\n".join(response_parts))
            
            update_dict["nudge_count"] = nudge_count + 1
            update_dict["agent_output"] = response_message
            messages.append(AIMessage(content=response_message))
            update_dict["messages"] = messages
    
    return update_dict


def _guided_logic(state: MathAgentState) -> Dict[str, Any]:
    """
    GUIDED mode: Acknowledge effort and provide targeted hint.
    
    - Point out what student understood correctly
    - Explicitly state what's missing
    - Provide clear hint toward solution
    """
    print("\n🧭 GUIDED MODE")
    
    messages = state.get("messages", [])
    problem = state["problem"]
    missing_concept = state.get("missing_concept")
    steps = state.get("steps", [])
    
    # Get user's latest response
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    
    # Build guided prompt with conversation history
    parser = PydanticOutputParser(pydantic_object=GuidedResponse)
    format_instructions = parser.get_format_instructions()
    
    step_context = "Full problem solving" if not steps else f"Working through {len(steps)} steps"
    missing_concept_info = f"Student may be missing: {missing_concept}" if missing_concept else "No specific concept gap detected"
    
    guided_user_msg = GUIDED_USER_TEMPLATE.format(
        problem=problem,
        step_context=step_context,
        user_input=user_input,
        missing_concept_info=missing_concept_info
    )
    
    # Build messages with conversation history
    guided_messages = build_messages_with_history(
        state=state,
        system_prompt=GUIDED_SYSTEM_PROMPT,
        user_prompt=guided_user_msg,
        format_instructions=format_instructions
    )
    
    # Invoke LLM
    print("🤖 Calling LLM for guided response...")
    response = invoke_llm_with_fallback(guided_messages, "GUIDED")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        guided_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing guided response: {e}")
        print(f"Raw response: {response.content}")
        raise RuntimeError("Failed to parse GUIDED response") from e
    
    # Build response message as a flowing, conversational paragraph
    # Rather than four separate blocks separated by newlines, weave them together
    # so the tutor sounds natural, not like a form being filled out.
    response_message = (
        f"{guided_resp.acknowledgment} "
        f"{guided_resp.missing_piece} "
        f"{guided_resp.hint} "
        f"{guided_resp.encouragement}"
    )
    translated_message = translate_if_kannada(state, response_message)
    
    messages.append(AIMessage(content=translated_message))
    
    return {
        "agent_output": translated_message,
        "messages": messages,
        "current_state": "ADAPTIVE_SOLVER",
    }


def _scaffold_logic(state: MathAgentState) -> Dict[str, Any]:
    """
    SCAFFOLD mode: Guide student step-by-step, checking each answer before advancing.

    State machine driven by scaffold_retry_count:
      - retry_count == 0  → First time on this step: present instruction + check question
      - retry_count >= 1  → Student answered: LLM evaluates and responds in same call
        - should_advance=True  → step_index+1, retry_count reset to 0
        - should_advance=False AND retry_count < MAX → retry_count+1, stay on step
        - should_advance=False AND retry_count >= MAX → reveal answer, advance step
    """
    print("\n🪜 SCAFFOLD MODE")

    messages = state.get("messages", [])
    problem = state["problem"]
    steps = state.get("steps", [])
    step_index = state.get("step_index", 0)
    retry_count = state.get("scaffold_retry_count", 0)

    # Get user's latest response (may be None on the very first scaffold call)
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    # ── All steps done ────────────────────────────────────────────────────────
    if step_index >= len(steps):
        print("✅ All steps completed!")
        completion_message = "You did it! You've worked through every single step — that's the complete solution! 🎉 I'm really proud of you."
        translated_message = translate_if_kannada(state, completion_message)
        messages.append(AIMessage(content=translated_message))
        return {
            "solved": True,
            "agent_output": translated_message,
            "messages": messages,
            "current_state": "ADAPTIVE_SOLVER",
        }

    current_step = steps[step_index]
    step_description = current_step.get("description", "")
    step_concept = current_step.get("concept", "general")

    # Determine mode: first instruction vs evaluation
    is_first_instruction = (retry_count == 0)

    print(f"📍 Step {step_index + 1}/{len(steps)} | retry_count={retry_count} | first={is_first_instruction}")

    # ── Max retries reached on this step: reveal and move on ─────────────────
    if not is_first_instruction and retry_count >= MAX_SCAFFOLD_RETRIES:
        print("⏭️ Max retries reached — revealing step answer")

        reveal_parser = PydanticOutputParser(pydantic_object=ScaffoldRevealResponse)
        reveal_format_instructions = reveal_parser.get_format_instructions()

        reveal_user_msg = SCAFFOLD_REVEAL_USER_TEMPLATE.format(
            problem=problem,
            step_index=step_index + 1,
            current_step=step_description,
            step_concept=step_concept,
            retry_count=retry_count,
        )

        reveal_messages = build_messages_with_history(
            state=state,
            system_prompt=SCAFFOLD_REVEAL_SYSTEM_PROMPT,
            user_prompt=reveal_user_msg,
            format_instructions=reveal_format_instructions,
        )

        print("🤖 Calling LLM for scaffold reveal...")
        reveal_response = invoke_llm_with_fallback(reveal_messages, "SCAFFOLD_REVEAL")

        try:
            json_str = extract_json_block(reveal_response.content)
            reveal_resp = reveal_parser.parse(json_str)
        except Exception as e:
            print(f"❌ Reveal parse error: {e}")
            print(f"Raw response: {reveal_response.content}")
            raise RuntimeError("Failed to parse SCAFFOLD_REVEAL response") from e

        answer_message = (
            f"{reveal_resp.acknowledgment} "
            f"{reveal_resp.reveal} "
            f"{reveal_resp.encouragement}"
        )
        translated_message = translate_if_kannada(state, answer_message)
        messages.append(AIMessage(content=translated_message))

        return {
            "step_index": step_index + 1,
            "scaffold_retry_count": 0,
            "agent_output": translated_message,
            "messages": messages,
            "current_state": "ADAPTIVE_SOLVER",
        }

    # ── Build scaffold prompt (handles first instruction OR evaluation) ────────
    parser = PydanticOutputParser(pydantic_object=ScaffoldResponse)
    format_instructions = parser.get_format_instructions()

    if is_first_instruction:
        is_first_str = "YES — introduce this step clearly for the first time and ask a check question.If the student has already solved this step correctly then you can directly move on after acknowledging it."
        retry_context = ""
    else:
        is_first_str = f"NO — the student has responded (attempt {retry_count} of {MAX_SCAFFOLD_RETRIES}). Look at their latest message in conversation history and evaluate it."
        retry_context = f"Attempts on this step so far: {retry_count} of {MAX_SCAFFOLD_RETRIES} max before I must reveal the answer."

    scaffold_user_msg = SCAFFOLD_USER_TEMPLATE.format(
        problem=problem,
        step_index=step_index + 1,
        total_steps=len(steps),
        current_step=step_description,
        step_concept=step_concept,
        is_first_instruction=is_first_str,
        retry_context=retry_context,
    )

    scaffold_messages = build_messages_with_history(
        state=state,
        system_prompt=SCAFFOLD_SYSTEM_PROMPT,
        user_prompt=scaffold_user_msg,
        format_instructions=format_instructions,
    )

    print("🤖 Calling LLM for scaffold response...")
    response = invoke_llm_with_fallback(scaffold_messages, "SCAFFOLD")

    try:
        json_str = extract_json_block(response.content)
        scaffold_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Parse error: {e}")
        print(f"Raw response: {response.content}")
        print("🔧 Attempting one JSON repair pass for SCAFFOLD response...")
        try:
            scaffold_resp = _repair_scaffold_json_response(
                state=state,
                raw_response=response.content,
                format_instructions=format_instructions,
            )
            print("✅ SCAFFOLD JSON repair succeeded")
        except Exception as repair_error:
            print(f"❌ SCAFFOLD JSON repair failed: {repair_error}")
            raise RuntimeError("Failed to parse SCAFFOLD response") from e

    translated_message = translate_if_kannada(state, scaffold_resp.response_to_student)
    messages.append(AIMessage(content=translated_message))

    print(f"   is_correct={scaffold_resp.is_correct} | should_advance={scaffold_resp.should_advance}")

    # ── Update state based on LLM decision ───────────────────────────────────
    update_dict = {
        "agent_output": translated_message,
        "messages": messages,
        "current_state": "ADAPTIVE_SOLVER",
        "current_step_description": step_description,
    }

    if is_first_instruction:
        # Presented instruction — now waiting for student's attempt
        update_dict["scaffold_retry_count"] = 1
    elif scaffold_resp.should_advance:
        # Student got it right — advance to next step
        print(f"✅ Step {step_index + 1} complete — advancing to step {step_index + 2}")
        update_dict["step_index"] = step_index + 1
        update_dict["scaffold_retry_count"] = 0
    else:
        # Wrong answer — retry
        update_dict["scaffold_retry_count"] = retry_count + 1
        print(f"🔄 Wrong answer — retry_count now {retry_count + 1}")

    return update_dict



# ============================================================================
# REFLECTION NODE
# ============================================================================

def reflection_node(state: MathAgentState) -> Dict[str, Any]:
    """
    REFLECTION node: Celebrate success and suggest next steps.
    
    - Appreciate student's effort and success
    - Check confidence level
    - Suggest meaningful next actions
    
    Returns:
        Partial state update with reflection message
    """
    print("\n" + "="*60)
    print("🎉 REFLECTION NODE")
    print("="*60)
    
    messages = state.get("messages", [])
    problem = state["problem"]
    mode = state.get("mode", "unknown")
    nudge_count = state.get("nudge_count", 0)
    step_index = state.get("step_index", 0)
    
    # Load problem data for final answer
    problem_id = state.get("problem_id")
    problem_data = load_problem_from_json(problem_id)
    final_answer = problem_data.get("final_answer", "the correct answer")
    steps = problem_data.get("canonical_solution", {}).get("steps", [])
    
    # Determine concepts learned
    concepts_learned = state.get("concepts_taught", [])
    
    # Build reflection prompt with conversation history
    parser = PydanticOutputParser(pydantic_object=ReflectionResponse)
    format_instructions = parser.get_format_instructions()
    
    reflection_user_msg = REFLECTION_USER_TEMPLATE.format(
        problem=problem,
        final_answer=final_answer,
        initial_mode=mode,
        concepts_learned=", ".join(concepts_learned) if concepts_learned else "None",
        attempt_count=nudge_count + step_index
    )
    
    # Build messages with conversation history
    reflection_messages = build_messages_with_history(
        state=state,
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        user_prompt=reflection_user_msg,
        format_instructions=format_instructions
    )
    
    # Invoke LLM
    print("🤖 Calling LLM for reflection...")
    response = invoke_llm_with_fallback(reflection_messages, "REFLECTION")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        reflection_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing reflection response: {e}")
        print(f"Raw response: {response.content}")
        raise RuntimeError("Failed to parse REFLECTION response") from e
    
    # Build response message
    response_parts = [
        reflection_resp.appreciation,
        f"\n{reflection_resp.confidence_check}",
        f"\n{reflection_resp.summary_offer}"
    ]
    
    response_message = "\n".join(response_parts)
    translated_message = translate_if_kannada(state, response_message)
    messages.append(AIMessage(content=translated_message))
    
    print("✅ Reflection complete")
    
    return {
        "agent_output": translated_message,
        "messages": messages,
        "current_state": "HANDLE_SUMMARY_REQUEST", # Route to summary request handler
    }


# ============================================================================
# HANDLE SUMMARY REQUEST NODE 
# ============================================================================

def handle_summary_request_node(state: MathAgentState) -> Dict[str, Any]:
    """
    HANDLE SUMMARY REQUEST node.
    
    - Evaluates student's response to the summary offer.
    - If yes, provides steps and asks what next.
    - If no, just asks what next.
    """
    print("\n" + "="*60)
    print("📝 HANDLE SUMMARY REQUEST NODE")
    print("="*60)

    messages = state.get("messages", [])
    problem_id = state.get("problem_id")
    
    # Get user's latest response
    user_input = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    # Build prompt
    parser = PydanticOutputParser(pydantic_object=HandleSummaryResponse)
    format_instructions = parser.get_format_instructions()
    
    handle_summary_user_msg = HANDLE_SUMMARY_REQUEST_USER_TEMPLATE.format(
        student_reply=user_input
    )
    
    handle_messages = build_messages_with_history(
        state=state,
        system_prompt=HANDLE_SUMMARY_REQUEST_SYSTEM_PROMPT,
        user_prompt=handle_summary_user_msg,
        format_instructions=format_instructions
    )
    
    print("🤖 Calling LLM for summary request handling...")
    response = invoke_llm_with_fallback(handle_messages, "HANDLE_SUMMARY_REQUEST")
    
    try:
        json_str = extract_json_block(response.content)
        handle_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing handle summary response: {e}")
        print(f"Raw response: {response.content}")
        raise RuntimeError("Failed to parse HANDLE_SUMMARY_REQUEST response") from e

    response_parts = [handle_resp.response_prefix]

    # If they want the summary, fetch and append the steps
    if handle_resp.wants_summary:
        problem_data = load_problem_from_json(problem_id)
        steps = problem_data.get("canonical_solution", {}).get("steps", [])
        if steps:
            response_parts.append("\n**Step-by-step summary:**\n")
            for i, step in enumerate(steps, 1):
                response_parts.append(f"**Step {i}:** {step.get('description', '')}")
            response_parts.append("\n---")
    
    # Add next action suggestions
    response_parts.append("\n**What would you like to do next?**")
    for i, suggestion in enumerate(handle_resp.next_action_suggestions, 1):
        response_parts.append(f"{i}. {suggestion}")

    response_message = "\n".join(response_parts)
    translated_message = translate_if_kannada(state, response_message)
    messages.append(AIMessage(content=translated_message))

    return {
        "agent_output": translated_message,
        "messages": messages,
        "current_state": "END"
    }

def end_node(state: MathAgentState) -> Dict[str, Any]:
    """
    END node: End the conversation.
    """
    print("\n" + "="*60)
    print("🏁 END NODE")
    print("="*60)
    
    return {
        "current_state": "END",
    }
