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
    ConceptCheckResponse,
    ApproachAssessmentResponse,
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
    CONCEPT_CHECK_SYSTEM_PROMPT,
    CONCEPT_CHECK_USER_TEMPLATE,
    RE_ASK_SYSTEM_PROMPT,
    RE_ASK_USER_TEMPLATE,
    APPROACH_ASSESSMENT_SYSTEM_PROMPT,
    APPROACH_ASSESSMENT_USER_TEMPLATE,
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
        # New concept tracking fields
        "missing_concepts": [],
        "concepts_taught": [],
        "concept_visit_count": {},
        "concept_interaction_count": 0,
        "post_concept_reassessment": False,
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
    concept_check_messages = build_messages_with_history(
        state=state,
        system_prompt=CONCEPT_CHECK_SYSTEM_PROMPT,
        user_prompt=concept_check_user_msg,
        format_instructions=format_instructions
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
        # Fallback to no missing concepts
        concept_check = ConceptCheckResponse(
            missing_concepts=[],
            reasoning="Unable to parse concept check response"
        )
    
    print(f"📊 Concept Check Results:")
    print(f"   Missing Concepts: {concept_check.missing_concepts or 'None'}")
    print(f"   Reasoning: {concept_check.reasoning}")
    
    if concept_check.missing_concepts:
        print(f"🎯 Routing to CONCEPT node (missing: {', '.join(concept_check.missing_concepts)})")
    else:
        print(f"✅ All concepts understood, routing to ASSESS_APPROACH")
    
    return {
        "missing_concepts": concept_check.missing_concepts,
        "last_user_msg": user_input,
        "current_state": "ASSESSMENT",
    }


# ============================================================================
# CONCEPT NODE (Standalone - Not a Mode)
# ============================================================================

def concept_node(state: MathAgentState) -> Dict[str, Any]:
    """
    CONCEPT node: Teach missing prerequisite concepts.
    
    Features:
    - Teaches concepts with analogies and examples
    - Tracks interaction count (max 3 per session)
    - Tracks visit count (max 2 per concept)
    - Asks micro-check questions to verify understanding
    
    Returns:
        Partial state update with concepts_taught and updated counters
    """
    print("\n" + "="*60)
    print("💡 CONCEPT NODE")
    print("="*60)
    
    messages = state.get("messages", [])
    problem = state["problem"]
    missing_concepts = state.get("missing_concepts", [])
    concepts_taught = state.get("concepts_taught", [])
    concept_visit_count = state.get("concept_visit_count", {})
    interaction_count = state.get("concept_interaction_count", 0)
    
    # Get user's latest response (if any - might be first time in concept node)
    user_input = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    
    if not missing_concepts:
        print("⚠️ No missing concepts specified, should not be in concept node")
        return {
            "current_state": "CONCEPT",
        }
    
    # Get the first concept to teach (we teach one at a time)
    current_concept = missing_concepts[0]
    
    # Check if we've exceeded visit limit for this concept
    visits = concept_visit_count.get(current_concept, 0)
    print(f"📚 Teaching concept: {current_concept} (visit {visits + 1}/{MAX_CONCEPT_VISITS_PER_CONCEPT})")
    
    # Check if we've exceeded interaction limit
    if interaction_count >= MAX_CONCEPT_INTERACTIONS:
        print(f"⏭️ Max interactions ({MAX_CONCEPT_INTERACTIONS}) reached for this concept session")
        # Move on - mark concept as taught even if not fully understood
        concepts_taught.append(current_concept)
        remaining_concepts = missing_concepts[1:]
        
        if remaining_concepts:
            # More concepts to teach
            print(f"📚 Moving to next concept. Remaining: {remaining_concepts}")
            return {
                "missing_concepts": remaining_concepts,
                "concepts_taught": concepts_taught,
                "concept_interaction_count": 0,  # Reset for next concept
                "current_state": "CONCEPT",
            }
        else:
            # All concepts taught (or attempted)
            print("✅ All concepts covered")
            return {
                "missing_concepts": [],
                "concepts_taught": concepts_taught,
                "concept_interaction_count": 0,
                "current_state": "CONCEPT",
            }
    
    # Build concept teaching prompt
    parser = PydanticOutputParser(pydantic_object=ConceptResponse)
    format_instructions = parser.get_format_instructions()
    
    concept_user_msg = CONCEPT_USER_TEMPLATE.format(
        missing_concept=current_concept,
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
    print(f"🤖 Calling LLM to teach concept: {current_concept}")
    response = invoke_llm_with_fallback(concept_messages, "CONCEPT")
    
    # Parse response
    try:
        json_str = extract_json_block(response.content)
        concept_resp = parser.parse(json_str)
    except Exception as e:
        print(f"❌ Error parsing concept response: {e}")
        # Fallback response
        concept_resp = ConceptResponse(
            concept_explanation=f"Let me explain {current_concept}...",
            analogy="Think of it like this...",
            micro_check_question="Do you understand?",
            encouragement="You're doing great!"
        )
    
    # Build response message
    response_message = f"**Let's learn about {current_concept}!** 🌟\n\n{concept_resp.concept_explanation}\n\n**Analogy:** {concept_resp.analogy}\n\n{concept_resp.encouragement}\n\n**Quick Check:** {concept_resp.micro_check_question}"
    
    messages.append(AIMessage(content=response_message))
    
    # Increment interaction count
    interaction_count += 1
    
    # Update visit count for this concept
    concept_visit_count[current_concept] = visits + 1
    
    # If student has responded to micro-check, move to next concept
    if user_input and interaction_count >= 1:
        # Simple heuristic: if they responded, assume they attempted the micro-check
        # Mark this concept as taught
        if current_concept not in concepts_taught:
            concepts_taught.append(current_concept)
        
        remaining_concepts = missing_concepts[1:]
        
        if remaining_concepts:
            # More concepts to teach
            print(f"📚 Moving to next concept. Remaining: {remaining_concepts}")
            return {
                "missing_concepts": remaining_concepts,
                "concepts_taught": concepts_taught,
                "concept_visit_count": concept_visit_count,
                "concept_interaction_count": 0,  # Reset for next concept
                "agent_output": response_message,
                "messages": messages,
                "current_state": "CONCEPT",
            }
        else:
            # All concepts taught
            print("✅ All concepts taught")
            return {
                "missing_concepts": [],
                "concepts_taught": concepts_taught,
                "concept_visit_count": concept_visit_count,
                "concept_interaction_count": 0,
                "agent_output": response_message,
                "messages": messages,
                "current_state": "CONCEPT",
            }
    
    return {
        "concept_visit_count": concept_visit_count,
        "concept_interaction_count": interaction_count,
        "agent_output": response_message,
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
    
    # Build re-ask prompt
    parser = PydanticOutputParser(pydantic_object=str)  # Simple string response
    
    re_ask_user_msg = RE_ASK_USER_TEMPLATE.format(
        problem=problem,
        concepts_taught=", ".join(concepts_taught) if concepts_taught else "some key concepts"
    )
    
    # Build messages
    re_ask_messages = [
        SystemMessage(content=RE_ASK_SYSTEM_PROMPT),
        HumanMessage(content=re_ask_user_msg)
    ]
    
    # Invoke LLM
    print(f"🤖 Calling LLM to re-ask questions after teaching: {concepts_taught}")
    response = invoke_llm_with_fallback(re_ask_messages, "RE_ASK")
    
    response_message = response.content
    messages.append(AIMessage(content=response_message))
    
    print("✅ Re-asked START questions")
    
    return {
        "agent_output": response_message,
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
        raise e
        # # Fallback to default values
        # assessment = ApproachAssessmentResponse(
        #     Tu=0.3,
        #     Ta=0.3,
        #     reasoning="Unable to parse assessment response"
        # )
    
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
