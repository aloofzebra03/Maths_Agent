# simulation_nodes.py
import json
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from utils.shared_utils import (
    AgentState,
    add_ai_message_to_conversation,
    llm_with_history,
    build_prompt_from_template,
    build_prompt_from_template_optimized,
    extract_json_block,
    create_simulation_config
)


# ─────────────────────────────────────────────────────────────────────
# Simulation moves
# ─────────────────────────────────────────────────────────────────────

SIM_MOVES: Dict[str, Dict[str, str]] = {
    "SIM_CC":     {"goal": "Propose 1–5 distinct, independently variable core concepts for simulation.", "constraints": "Clear, learner-friendly; testable via observable changes."},
    "SIM_VARS":   {"goal": "List variables; mark independent/dependent/controls.", "constraints": "Only relevant variables; concise."},
    "SIM_ACTION": {"goal": "Describe a single, testable change to perform.", "constraints": "Alter one independent variable; keep controls fixed."},
    "SIM_EXPECT": {"goal": "Elicit the learner's prediction and brief why.", "constraints": "No leading hints; accept 'not sure'."},
    "SIM_EXECUTE":{"goal": "Perform/narrate the action steps.", "constraints": "Describe observable effects only; no interpretation."},
    "SIM_OBSERVE":{"goal": "Ask for raw observations from the learner.", "constraints": "Allow multiple valid answers; don't judge yet."},
    "SIM_INSIGHT":{"goal": "Map observation → principle; compare with prediction.", "constraints": "Reinforce why it happened; ≤3 sentences."},
    "SIM_REFLECT":{"goal": "Synthesize learning across the simulated concept(s).", "constraints": "Encourage metacognition; concise bullets."},
}

# ─────────────────────────────────────────────────────────────────────
# Pydantic response models + parsers (same style as main nodes)
# ─────────────────────────────────────────────────────────────────────

# SIM_CC: 1–5 concepts
class SimConcepts(BaseModel):
    concepts: List[str] = Field(description="List of clear, testable concepts that can be demonstrated to the student", min_items=1, max_items=5)


# SIM_VARS: declare variables
class SimVariable(BaseModel):
    name: str
    role: Literal["independent", "dependent", "control"]
    note: Optional[str] = None

class SimVarsResponse(BaseModel):
    variables: List[SimVariable] = Field(description="List of variables for the simulation", min_items=2)
    prompt_to_learner: str = Field(description="Direct question or statement to the student. Use 'you' and address them personally. Avoid third-person references like 'the student'.")

# SIM_ACTION
class SimActionResponse(BaseModel):
    action: str = Field(description="Describe the action in clear terms that can be communicated directly to the student. Build on the variables just discussed.")
    rationale: str = Field(description="Explain to the student why this action will help them understand the concept. Reference the variables they just learned about.")
    prompt_to_learner: str = Field(description="Direct question to the student asking for their consent or engagement. Use 'you' and speak directly to them.Also respond to previous user response")

# SIM_EXPECT
class SimExpectResponse(BaseModel):
    question: str = Field(description="Direct question asking the student for their prediction about the proposed action. Use 'you' and 'your' - speak directly to the student.")
    hint: Optional[str] = Field(default=None, description="Optional gentle hint for the student, phrased as if speaking directly to them")

# SIM_EXECUTE
class SimExecuteResponse(BaseModel):
    steps: List[str] = Field(description="List of steps describing what you are doing for the student to observe", min_items=1)
    what_to_watch: str = Field(description="Tell the student directly what they should focus on watching during the simulation")

# SIM_OBSERVE
class SimObserveResponse(BaseModel):
    observation_prompt: str = Field(description="Direct question asking the student what they observed. Use 'you' and speak directly to them.")
    expected_observations: List[str] = Field(description="Internal list of what the student might observe - not shown verbatim to student")

# SIM_INSIGHT
class SimInsightResponse(BaseModel):
    micro_explanation: str = Field(description="Explain to the student why they observed what they did. Speak directly to them using 'you' and 'your'.")
    compared_to_prediction: str = Field(description="Connect directly with the student about their prediction using 'you' and 'your prediction'.")

# SIM_REFLECT
class SimReflectResponse(BaseModel):
    bullets: List[str] = Field(description="Key takeaways phrased as if speaking directly to the student", min_items=2, max_items=5)
    closing_prompt: str = Field(description="Reflective question for the student using direct address with 'you' and 'your'.")

sim_cc_parser = PydanticOutputParser(pydantic_object=SimConcepts)
sim_vars_parser = PydanticOutputParser(pydantic_object=SimVarsResponse)
sim_action_parser = PydanticOutputParser(pydantic_object=SimActionResponse)
sim_expect_parser = PydanticOutputParser(pydantic_object=SimExpectResponse)
sim_execute_parser = PydanticOutputParser(pydantic_object=SimExecuteResponse)
sim_observe_parser = PydanticOutputParser(pydantic_object=SimObserveResponse)
sim_insight_parser = PydanticOutputParser(pydantic_object=SimInsightResponse)
sim_reflect_parser = PydanticOutputParser(pydantic_object=SimReflectResponse)

def sim_concept_creator_node(state: AgentState) -> AgentState:
    """
    SIM_CC: Generate 1–5 independently variable, testable concepts.
    Stores:
      - sim_concepts: List[str]
      - sim_total_concepts: int
      - sim_current_idx: int
    Handover: set current_state="GE" so graph routes to your GE node.
    """

    context = json.dumps(SIM_MOVES["SIM_CC"], indent=2)
    system_prompt = f"""Current node: SIM_CC (Simulation Concept Creator)

You are talking directly with a class 7 student. Remember to use direct communication.

Concept in focus: "{state['concept_title']}"
When the concept in focus is Simple Pendulum or pendulum and its Time Period focus ONLY on the Length,Amplitude and Mass of the pendulum.NOTHING else.

Context:
{context}

Task:
Return JSON ONLY with 1–5 clear, independent, testable simulation concepts for a class 7 learner.

Guidelines:
- Keep each concept short (≤15 words), concrete, and experimentally manipulable.
- They must be independently variable (changing one doesn't implicitly change the others).
"""

    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=False,
        include_instructions=True,
        parser=sim_cc_parser,
        current_node="SIM_CC"
    )
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    parsed: SimConcepts = sim_cc_parser.parse(json_text)

    # Save & speak
    state["sim_concepts"] = parsed.concepts
    state["sim_total_concepts"] = len(parsed.concepts)
    state["sim_current_idx"] = 0 

    speak = (
        "We'll explore these concepts together, one by one:\n"
        + "\n".join([f"{i+1}. {c}" for i, c in enumerate(parsed.concepts)])
        + f"\n\nLet's start with the first concept: '{parsed.concepts[0]}'. Are you ready?"
    )
    add_ai_message_to_conversation(state, speak)
    state["agent_output"] = speak

    state["current_state"] = "GE"
    return state


def sim_vars_node(state: AgentState) -> AgentState:
    """
    SIM_VARS: List variables (independent/dependent/control) for current concept.
    """
    idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    concept = concepts[idx]

    context = json.dumps(SIM_MOVES["SIM_VARS"], indent=2)
    system_prompt = f"""Current node: SIM_VARS (Variables Declaration)

You are talking directly with a class 7 student. Remember to address them directly using 'you' and avoid third-person references like 'the student'.

We are on Simulation Concept #{idx+1}: "{concept}"

Context:
{context}

Task:
Respond with JSON ONLY: declare variables (independent/dependent/control) that matter to this concept, and a short prompt to the learner to confirm/ask questions.

Keep it concise and age-appropriate. Address the student directly in your prompt_to_learner.
"""

    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=sim_vars_parser,
        current_node="SIM_VARS"
    )
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    parsed: SimVarsResponse = sim_vars_parser.parse(json_text)

    lines = ["Here are the variables we'll work with:"]
    for v in parsed.variables:
        note = f" — {v.note}" if v.note else ""
        lines.append(f"- {v.name} ({v.role}){note}")
    lines.append(parsed.prompt_to_learner)
    msg = "\n".join(lines)

    # Store variables for later use in simulation - convert Pydantic objects to dictionaries
    state["sim_variables"] = [
        {
            "name": v.name,
            "role": v.role,
            "note": v.note
        }
        for v in parsed.variables
    ]

    add_ai_message_to_conversation(state, msg)
    state["agent_output"] = msg
    # state["current_state"] = "SIM_ACTION"
    state["current_state"] = "AR" #No simulation agent for now
    return state


def sim_action_node(state: AgentState) -> AgentState:
    """
    SIM_ACTION: Propose one concrete action to isolate the concept.
    """
    idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    concept = concepts[idx]

    context = json.dumps(SIM_MOVES["SIM_ACTION"], indent=2)
    system_prompt = f"""Current node: SIM_ACTION (Single Manipulation)

You are talking directly with a class 7 student. Use direct communication and address them as 'you'. 
The student has just learned about the variables we'll be working with.

We are on Simulation Concept #{idx+1}: "{concept}"

Context:
{context}

Task:
Return JSON ONLY describing:
- 'action': one specific manipulation on the independent variable (explain what you will do)
- 'rationale': why this will help the student understand the concept (speak directly to them)
- 'prompt_to_learner': a direct question asking for their engagement (use 'you')In this itself also acknowledge what the user said before and respond accordingly.
"""

    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=sim_action_parser,
        current_node="SIM_ACTION"
    )
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    parsed: SimActionResponse = sim_action_parser.parse(json_text)

    msg = f"{parsed.action}\n\nWhy this works: {parsed.rationale}\n{parsed.prompt_to_learner}"
    
    # Store action configuration for simulation
    state["sim_action_config"] = {
        "action": parsed.action,
        "rationale": parsed.rationale,
        "prompt": parsed.prompt_to_learner
    }
    
    add_ai_message_to_conversation(state, msg)
    state["agent_output"] = msg
    state["current_state"] = "SIM_EXPECT"
    return state


def sim_expect_node(state: AgentState) -> AgentState:
    """
    SIM_EXPECT: Ask the learner's prediction before executing.
    """
    idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    concept = concepts[idx]

    context = json.dumps(SIM_MOVES["SIM_EXPECT"], indent=2)
    system_prompt = f"""Current node: SIM_EXPECT (Prediction)

You are talking directly with a class 7 student. Use 'you' and 'your' when asking for their prediction.
The student has just agreed to try the proposed action/experiment.

We are on Simulation Concept #{idx+1}: "{concept}"

Context:
{context}

Task:
Return JSON ONLY with:
- 'question': Ask the student directly for their prediction using 'you' and 'your'
- 'hint': optional gentle hint phrased as if speaking directly to the student (or null)
"""

    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=sim_expect_parser,
        current_node="SIM_EXPECT"
    )
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    parsed: SimExpectResponse = sim_expect_parser.parse(json_text)

    hint = f"\n(Hint: {parsed.hint})" if parsed.hint else ""
    msg = f"{parsed.question}{hint}"
    add_ai_message_to_conversation(state, msg)
    state["agent_output"] = msg
    state["current_state"] = "SIM_EXECUTE"
    return state


def sim_execute_node(state: AgentState) -> AgentState:
    """
    SIM_EXECUTE: Execute the simulation action with visual demonstration.
    Creates simulation config and sets flags for Streamlit to display the pendulum simulation.
    """
    # Get stored data from previous nodes
    variables = state.get("sim_variables", [])
    action_config = state.get("sim_action_config", {})
    idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    current_concept = concepts[idx] if idx < len(concepts) else "Unknown concept"
    
    # Create simulation configuration
    simulation_config = create_simulation_config(variables, current_concept, action_config)
    
    # Set flags for Streamlit to display simulation - but mark that it's active
    state["show_simulation"] = True
    state["simulation_config"] = simulation_config

    # flag = state.get("show_simulation",False)
    # state["show_simulation"] = not flag  # New flag to track simulation lifecycle
    
    # Agent message
    msg = f"Perfect! Let me demonstrate this concept with a simulation for you. {simulation_config['agent_message']}"
    # msg = f"Perfect! Let me demonstrate this concept with a simulation for you."
    add_ai_message_to_conversation(state, msg)
    state["agent_output"] = msg
    state["current_state"] = "SIM_OBSERVE"
    return state


def sim_observe_node(state: AgentState) -> AgentState:
    """
    SIM_OBSERVE: Ask for raw observations (no judging).
    """
    idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    concept = concepts[idx]

    context = json.dumps(SIM_MOVES["SIM_OBSERVE"], indent=2)
    system_prompt = f"""Current node: SIM_OBSERVE (What did you notice?)

You are talking directly with a class 7 student. Ask them directly what they observed using 'you'.
The student has just watched the simulation demonstration.

We are on Simulation Concept #{idx+1}: "{concept}"

Context:
{context}

Task:
Return JSON ONLY with:
- 'observation_prompt': Ask the student directly what they observed using 'you'
- 'expected_observations': 2–5 strings (internal guide, not printed verbatim later)
"""

    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=sim_observe_parser,
        current_node="SIM_OBSERVE"
    )
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    parsed: SimObserveResponse = sim_observe_parser.parse(json_text)

    add_ai_message_to_conversation(state, parsed.observation_prompt)
    state["agent_output"] = parsed.observation_prompt
    state["sim_expected_observations"] = parsed.expected_observations
    state["current_state"] = "SIM_INSIGHT"
    return state


def sim_insight_node(state: AgentState) -> AgentState:
    """
    SIM_INSIGHT: Micro-explanation + compare to prediction.
    (Graph will route to SIM_REFLECT next.)
    """
    idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    concept = concepts[idx]

    context = json.dumps(SIM_MOVES["SIM_INSIGHT"], indent=2)
    system_prompt = f"""Current node: SIM_INSIGHT (Why did that happen?)

You are talking directly with a class 7 student. Explain to them directly using 'you' and 'your'.
The student has just shared their observations from the simulation.

We are on Simulation Concept #{idx+1}: "{concept}"

Context:
{context}

Task:
Return JSON ONLY with:
- 'micro_explanation': ≤3 sentences explaining to the student why they observed what they did (use 'you')
- 'compared_to_prediction': 1 sentence connecting to the student's prediction using 'your prediction'
"""

    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=sim_insight_parser,
        current_node="SIM_INSIGHT"
    )
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    parsed: SimInsightResponse = sim_insight_parser.parse(json_text)

    msg = f"{parsed.micro_explanation}\n{parsed.compared_to_prediction}"
    add_ai_message_to_conversation(state, msg)
    state["agent_output"] = msg

    state["current_state"] = "SIM_REFLECT"
    return state


def sim_reflection_node(state: AgentState) -> AgentState:
    """
    SIM_REFLECT: Short synthesis across sim concept(s).
    Handover: set current_state="AR" to ask question about the concept.
    """
    idx = state.get("sim_current_idx", 0)
    concepts = state.get("sim_concepts", [])
    current_concept = concepts[idx] if concepts and idx < len(concepts) else "current concept"

    context = json.dumps(SIM_MOVES["SIM_REFLECT"], indent=2)
    system_prompt = f"""Current node: SIM_REFLECT (Synthesis)

You are talking directly with a class 7 student. Address them directly using 'you' and 'your'.
We have just completed the full simulation cycle and the student understands the insights.

We just completed a simulation for: "{current_concept}"

Context:
{context}

Task:
Return JSON ONLY with:
- 'bullets': 2–5 concise takeaways from this simulation, phrased as if speaking directly to the student
- 'closing_prompt': a short reflective question to the student about what they learned (use 'you')
"""

    final_prompt = build_prompt_from_template_optimized(
        system_prompt=system_prompt,
        state=state,
        include_last_message=True,
        include_instructions=True,
        parser=sim_reflect_parser,
        current_node="SIM_REFLECT"
    )
    raw = llm_with_history(state, final_prompt).content
    json_text = extract_json_block(raw)
    parsed: SimReflectResponse = sim_reflect_parser.parse(json_text)

    msg = f"Quick recap from our simulation:\n" + "\n".join([f"• {b}" for b in parsed.bullets]) + f"\n\n{parsed.closing_prompt}"
    add_ai_message_to_conversation(state, msg)
    state["agent_output"] = msg

    # # IMPORTANT: Reset simulation flags since simulation cycle is complete
    state["show_simulation"] = False
    state["simulation_config"] = {}
    
    # # Handover to AR to ask a question about this concept
    # # AR will handle concept progression and move to next concept via GE
    state["current_state"] = "AR"
    state["asked_ar"] = False  # Reset AR flag for this concept
    return state
