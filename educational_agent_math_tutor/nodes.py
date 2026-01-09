"""
Node implementations for the Math Tutoring Agent.

Contains all pedagogical nodes: START, ASSESSMENT, ADAPTIVE_SOLVER, REFLECTION.
"""

import json
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from educational_agent_math_tutor.schemas import (
    MathAgentState,
    AssessmentResponse,
    CoachResponse,
    GuidedResponse,
    ScaffoldResponse,
    ConceptResponse,
    ReflectionResponse,
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
    CONCEPT_SYSTEM_PROMPT,
    CONCEPT_USER_TEMPLATE,
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_USER_TEMPLATE,
)
from educational_agent_math_tutor.config import (
    TA_THRESHOLD_HIGH,
    TU_THRESHOLD_HIGH,
    MAX_COACH_NUDGES,
    MAX_SCAFFOLD_RETRIES,
)
from utils.shared_utils import (
    invoke_llm_with_fallback,
    extract_json_block,
    load_problem_from_json,
    format_required_concepts,
    build_messages_with_history,
)


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
    
    state['problem_id'] = 'add_frac_same_den_01'
    problem_id = state.get("problem_id")
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
    
    # Build system message with tutor persona
    messages = [
        SystemMessage(content=START_SYSTEM_PROMPT),
        AIMessage(content=greeting)
    ]
    
    print(f"✅ Loaded problem: {problem_id}")
    print(f"📝 Question: {question}")
    print(f"📊 Steps: {len(steps)}")
    print(f"🎯 Required concepts: {format_required_concepts(required_concepts)}")
    
    return {
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
    }


# ============================================================================
# ASSESSMENT NODE
# ============================================================================

def assess_student_response(state: MathAgentState) -> Dict[str, Any]:
    """
    ASSESSMENT node: Evaluate student's understanding (Tu) and approach (Ta).
    
    Uses LLM-as-grader with rubric to score:
    - Tu: Understanding of problem (0-1)
    - Ta: Quality of approach (0-1)
    - missing_concept: Detected prerequisite gap (if any)
    
    Routes to mode based on scores:
    - missing_concept → "concept"
    - Ta ≥ 0.6 AND Tu ≥ 0.6 → "coach"
    - Ta < 0.6 AND Tu < 0.6 → "scaffold"
    - else → "guided"
    
    Returns:
        Partial state update with Ta, Tu, mode, missing_concept
    """
    print("\n" + "="*60)
    print("📊 ASSESSMENT NODE")
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
    
    # Build assessment prompt with conversation history
    parser = PydanticOutputParser(pydantic_object=AssessmentResponse)
    format_instructions = parser.get_format_instructions()
    
    assessment_user_msg = ASSESSMENT_USER_TEMPLATE.format(
        problem=problem,
        user_input=user_input,
        required_concepts=required_concepts
    )
    
    # Build messages with conversation history
    assessment_messages = build_messages_with_history(
        state=state,
        system_prompt=ASSESSMENT_SYSTEM_PROMPT,
        user_prompt=assessment_user_msg,
        format_instructions=format_instructions
    )
    
    # Invoke LLM
    print("🤖 Calling LLM for assessment...")
    response = invoke_llm_with_fallback(assessment_messages, "ASSESSMENT")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        assessment = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing assessment response: {e}")
        print(f"Raw response: {response.content}")
        # Fallback to default values
        assessment = AssessmentResponse(
            Tu=0.3,
            Ta=0.3,
            reasoning="Unable to parse assessment response",
            missing_concept=None
        )
    
    print(f"📊 Assessment Results:")
    print(f"   Tu (Understanding): {assessment.Tu:.2f}")
    print(f"   Ta (Approach): {assessment.Ta:.2f}")
    print(f"   Reasoning: {assessment.reasoning}")
    print(f"   Missing Concept: {assessment.missing_concept or 'None'}")
    
    # Determine mode based on assessment
    if assessment.missing_concept:
        mode = "concept"
        print(f"🎯 Routing to CONCEPT mode (missing: {assessment.missing_concept})")
    elif assessment.Ta >= TA_THRESHOLD_HIGH and assessment.Tu >= TU_THRESHOLD_HIGH:
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
        "missing_concept": assessment.missing_concept,
        "last_user_msg": user_input,
        "current_state": "ASSESSMENT",
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
    - _concept_logic() for concept mode
    
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
    elif mode == "concept":
        return _concept_logic(state)
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
        # Fallback response
        coach_resp = CoachResponse(
            validation="Let me think about your approach.",
            is_correct=False,
            reflective_question="Can you explain your thinking?",
            encouragement="You're doing great!"
        )
    
    # Build response message
    response_parts = [coach_resp.validation]
    
    update_dict = {
        "current_state": "ADAPTIVE_SOLVER",
    }
    
    if coach_resp.is_correct:
        print("✅ Student answer is correct!")
        response_parts.append(coach_resp.encouragement)
        response_message = "\n\n".join(response_parts)
        
        update_dict["solved"] = True
        update_dict["agent_output"] = response_message
        messages.append(AIMessage(content=response_message))
        update_dict["messages"] = messages
        
    else:
        print(f"❌ Student answer incorrect (nudge {nudge_count + 1}/{MAX_COACH_NUDGES})")
        
        if nudge_count >= MAX_COACH_NUDGES:
            print("⬇️ Max nudges reached, downgrading to GUIDED mode")
            response_parts.append("I can see you're working hard on this. Let me give you some more specific help.")
            response_message = "\n\n".join(response_parts)
            
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
            response_message = "\n\n".join(response_parts)
            
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
        # Fallback response
        guided_resp = GuidedResponse(
            acknowledgment="I see you're thinking about this.",
            missing_piece="We need to focus on the method.",
            hint="Think about what operation you need to do.",
            encouragement="You're on the right track!"
        )
    
    # Build response message
    response_message = f"{guided_resp.acknowledgment}\n\n{guided_resp.missing_piece}\n\n{guided_resp.hint}\n\n{guided_resp.encouragement}"
    
    messages.append(AIMessage(content=response_message))
    
    return {
        "agent_output": response_message,
        "messages": messages,
        "current_state": "ADAPTIVE_SOLVER",
    }


def _scaffold_logic(state: MathAgentState) -> Dict[str, Any]:
    """
    SCAFFOLD mode: Provide one concrete operation per step.
    
    - Give explicit instruction for current step
    - Check understanding before moving to next step
    - If student fails MAX_SCAFFOLD_RETRIES times on same step:
      - Give them the answer for that step
      - Move to next step
    """
    print("\n🪜 SCAFFOLD MODE")
    
    messages = state.get("messages", [])
    problem = state["problem"]
    steps = state.get("steps", [])
    step_index = state.get("step_index", 0)
    retry_count = state.get("scaffold_retry_count", 0)
    
    # Get user's latest response
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    
    # Check if we've completed all steps
    if step_index >= len(steps):
        print("✅ All steps completed!")
        completion_message = "Excellent work! You've completed all the steps. You solved the problem! 🎉"
        messages.append(AIMessage(content=completion_message))
        return {
            "solved": True,
            "agent_output": completion_message,
            "messages": messages,
            "current_state": "ADAPTIVE_SOLVER",
        }
    
    current_step = steps[step_index]
    step_description = current_step.get("description", "")
    step_concept = current_step.get("concept", "general")
    
    print(f"📍 Current step: {step_index + 1}/{len(steps)}")
    print(f"📝 Step: {step_description}")
    print(f"🔄 Retry count: {retry_count}/{MAX_SCAFFOLD_RETRIES}")
    
    # Check if we should give the answer and move on
    if user_input and retry_count >= MAX_SCAFFOLD_RETRIES:
        print("⏭️ Max retries reached, giving answer and moving to next step")
        
        answer_message = f"That's okay! Let me help you with this step.\n\n**Step {step_index + 1}:** {step_description}\n\nLet's move on to the next step!"
        messages.append(AIMessage(content=answer_message))
        
        return {
            "step_index": step_index + 1,
            "scaffold_retry_count": 0,  # Reset for next step
            "agent_output": answer_message,
            "messages": messages,
            "current_state": "ADAPTIVE_SOLVER",
        }
    
    # Build scaffold prompt with conversation history
    parser = PydanticOutputParser(pydantic_object=ScaffoldResponse)
    format_instructions = parser.get_format_instructions()
    
    scaffold_user_msg = SCAFFOLD_USER_TEMPLATE.format(
        problem=problem,
        step_index=step_index + 1,
        current_step=step_description,
        step_concept=step_concept,
        retry_count=retry_count,
        max_retries=MAX_SCAFFOLD_RETRIES
    )
    
    # Build messages with conversation history
    scaffold_messages = build_messages_with_history(
        state=state,
        system_prompt=SCAFFOLD_SYSTEM_PROMPT,
        user_prompt=scaffold_user_msg,
        format_instructions=format_instructions
    )
    
    # Invoke LLM
    print("🤖 Calling LLM for scaffold response...")
    response = invoke_llm_with_fallback(scaffold_messages, "SCAFFOLD")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        scaffold_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing scaffold response: {e}")
        # Fallback response
        scaffold_resp = ScaffoldResponse(
            step_instruction=step_description,
            step_context="Let's work on this step.",
            check_question="Can you do this step?"
        )
    
    # Build response message
    response_parts = [
        f"**Step {step_index + 1}:** {scaffold_resp.step_context}",
        f"\n{scaffold_resp.step_instruction}"
    ]
    
    if scaffold_resp.check_question:
        response_parts.append(f"\n{scaffold_resp.check_question}")
    
    response_message = "\n".join(response_parts)
    messages.append(AIMessage(content=response_message))
    
    # Update retry count if this is a retry (user has already responded)
    update_dict = {
        "agent_output": response_message,
        "messages": messages,
        "current_state": "ADAPTIVE_SOLVER",
        "current_step_description": step_description,
    }
    
    if user_input:
        # This is a retry on the same step
        update_dict["scaffold_retry_count"] = retry_count + 1
    
    return update_dict


def _concept_logic(state: MathAgentState) -> Dict[str, Any]:
    """
    CONCEPT mode: Teach missing prerequisite concept.
    
    - Explain concept using age-appropriate analogy
    - Provide one micro-check question
    - After correct answer to check: resume previous_mode
    """
    print("\n💡 CONCEPT MODE")
    
    messages = state.get("messages", [])
    problem = state["problem"]
    missing_concept = state.get("missing_concept")
    previous_mode = state.get("previous_mode")
    
    # Get user's latest response
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    
    if not missing_concept:
        print("⚠️ No missing concept specified, returning to previous mode")
        return {
            "mode": previous_mode or "guided",
            "current_state": "ADAPTIVE_SOLVER",
        }
    
    # Save current mode as previous_mode if not already set
    if not previous_mode:
        previous_mode = "guided"  # Default to guided after concept teaching
        update_previous_mode = True
    else:
        update_previous_mode = False
    
    # Build concept teaching prompt with conversation history
    parser = PydanticOutputParser(pydantic_object=ConceptResponse)
    format_instructions = parser.get_format_instructions()
    
    concept_user_msg = CONCEPT_USER_TEMPLATE.format(
        missing_concept=missing_concept,
        problem=problem
    )
    
    # Build messages with conversation history
    concept_messages = build_messages_with_history(
        state=state,
        system_prompt=CONCEPT_SYSTEM_PROMPT,
        user_prompt=concept_user_msg,
        format_instructions=format_instructions
    )
    
    # Invoke LLM
    print(f"🤖 Calling LLM to teach concept: {missing_concept}")
    response = invoke_llm_with_fallback(concept_messages, "CONCEPT")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        concept_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing concept response: {e}")
        # Fallback response
        concept_resp = ConceptResponse(
            concept_explanation=f"Let me explain {missing_concept}...",
            analogy="Think of it like this...",
            micro_check_question="Do you understand?",
            encouragement="You're doing great!"
        )
    
    # Build response message
    response_message = f"**Let's learn about {missing_concept}!**\n\n{concept_resp.concept_explanation}\n\n**Analogy:** {concept_resp.analogy}\n\n{concept_resp.encouragement}\n\n**Quick Check:** {concept_resp.micro_check_question}"
    
    messages.append(AIMessage(content=response_message))
    
    update_dict = {
        "agent_output": response_message,
        "messages": messages,
        "current_state": "ADAPTIVE_SOLVER",
    }
    
    if update_previous_mode:
        update_dict["previous_mode"] = previous_mode
    
    # After teaching, check if student answered the micro-check
    # For simplicity, we'll resume to previous mode immediately
    # In a more sophisticated version, you'd validate the micro-check answer first
    if user_input:  # If student has responded to micro-check
        print(f"✅ Concept taught, resuming to {previous_mode} mode")
        update_dict["mode"] = previous_mode
        update_dict["missing_concept"] = None  # Clear the missing concept
    
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
    
    # Determine concepts learned (if any)
    concepts_learned = []
    if state.get("missing_concept"):
        concepts_learned.append(state["missing_concept"])
    
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
        # Fallback response
        reflection_resp = ReflectionResponse(
            appreciation="Great job solving this problem!",
            confidence_check="How confident do you feel about this topic now?",
            next_action_suggestions=[
                "Try a similar problem",
                "Take a break - you've earned it!"
            ]
        )
    
    # Build response message
    response_parts = [
        reflection_resp.appreciation,
        f"\n{reflection_resp.confidence_check}",
        "\n**What would you like to do next?**"
    ]
    
    for i, suggestion in enumerate(reflection_resp.next_action_suggestions, 1):
        response_parts.append(f"{i}. {suggestion}")
    
    response_message = "\n".join(response_parts)
    messages.append(AIMessage(content=response_message))
    
    print("✅ Reflection complete")
    
    return {
        "agent_output": response_message,
        "messages": messages,
        "current_state": "REFLECTION",
    }
