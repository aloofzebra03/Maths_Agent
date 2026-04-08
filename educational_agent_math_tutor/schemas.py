"""
Pydantic schemas and state definition for the Math Tutoring Agent.
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# Type alias for pedagogical modes (concept is now a standalone node, not a mode)
Mode = Literal["coach", "guided", "scaffold"]


class MathAgentState(TypedDict, total=False):
    """
    State for the Math Tutoring Agent.
    
    Core Fields:
        - problem: The math problem text
        - problem_id: Identifier for loading from JSON
        - user_input: Latest student response
        - Ta: Approach quality score (0-1)
        - Tu: Understanding quality score (0-1)
        - mode: Current pedagogical mode
        - solved: Whether problem is complete
        
    Step Tracking:
        - steps: List of solution steps from JSON (with concept info)
        - step_index: Current step in scaffold mode
        - max_steps: Total number of steps
        - current_step_description: Description of current step
        
    Pedagogical Context:
        - missing_concept: Detected prerequisite gap
        - previous_mode: Mode to resume after concept teaching
        - nudge_count: Number of reflective questions asked in coach mode
        - scaffold_retry_count: Failed attempts on current scaffold step
        
    Message & Tracking:
        - messages: Conversation history (annotated with add_messages)
        - current_state: Current node name
        - last_user_msg: Last message from student
        - agent_output: Last agent response
        - node_transitions: List of transitions with timestamps
        
    Future Optimization (unused for now):
        - summary: Rolling conversation summary
        - summary_last_index: Last summarized message index
    """
    
    # Core problem fields
    problem: str
    problem_id: str
    
    # Scoring
    Ta: float  # Approach quality (0-1)
    Tu: float  # Understanding quality (0-1)
    
    # Mode & state
    mode: Mode
    solved: bool
    is_kannada: Optional[bool]
    
    # Step tracking
    steps: List[Dict[str, Any]]  # Full step objects with step_id, description, concept
    step_index: int
    max_steps: int
    current_step_description: Optional[str]
    
    # Pedagogical context
    missing_concept: Optional[str]  # DEPRECATED: use missing_concepts instead
    previous_mode: Optional[Mode]
    nudge_count: int
    scaffold_retry_count: int
    
    # Concept teaching (new)
    missing_concepts: List[str]  # Concepts student doesn't know yet
    concepts_taught: List[str]  # Concepts already taught
    concept_visit_count: Dict[str, int]  # Track visits per concept {"denominator": 1}
    concept_interaction_count: int  # Interactions in current concept session
    post_concept_reassessment: bool  # Flag: have we re-asked after teaching?
    asked_concept: bool  # Flag: have we presented the concept teaching?
    concept_tries: int  # Number of student attempts at micro-check (max 3)

    # Start-node answer check
    start_attempt_count: int  # How many times student tried at start (0, 1, 2)
    awaiting_step_explanation: bool  # True when correct at start and waiting for step-explanation pref
    wants_step_explanation: Optional[bool]  # Student's choice: True=wants steps, False=skip
    final_answer: Optional[str]  # Correct answer stored for LLM context (never shown to student)

    
    # Message tracking
    messages: Annotated[List[AnyMessage], add_messages]
    current_state: str
    last_user_msg: Optional[str]
    agent_output: Optional[str]
    node_transitions: List[Dict[str, Any]]  # {timestamp, to_node, message_index}
    
    # Future optimization fields (unused for now)
    summary: Optional[str]
    summary_last_index: Optional[int]


# ============================================================================
# Pydantic Response Models for Structured LLM Outputs
# ============================================================================

class AssessmentResponse(BaseModel):
    """
    Structured response from the ASSESSMENT node.
    
    The LLM evaluates student's understanding and approach using a rubric,
    and detects any missing prerequisite concepts.
    """
    
    Tu: float = Field(
        description="Understanding score (0-1). Criteria: identifies operation needed, understands problem terms/meaning, knows what result represents",
        ge=0.0,
        le=1.0
    )
    
    Ta: float = Field(
        description="Approach score (0-1). Criteria: mentions correct method, logical step order, handles conversion/edge cases",
        ge=0.0,
        le=1.0
    )
    
    reasoning: str = Field(
        description="Brief explanation of the scores and what the student understands vs. struggles with"
    )
    
    missing_concept: Optional[str] = Field(
        default=None,
        description="If a prerequisite concept is missing (e.g., 'denominator', 'equivalent_fractions'), name it here. Otherwise null."
    )
    
    @field_validator('Tu', 'Ta')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {v}")
        return v


class CoachResponse(BaseModel):
    """
    Response from COACH mode - validate student work and ask reflective questions.
    """
    
    validation: str = Field(
        description="Brief validation of student's work. Praise effort even if incorrect."
    )
    
    is_correct: bool = Field(
        description="Whether the student's answer/approach is correct"
    )
    
    reflective_question: Optional[str] = Field(
        default=None,
        description="If incorrect, ask a 'why' question to guide reflection (e.g., 'Why did you add the denominators?'). Max 3 nudges."
    )
    
    encouragement: str = Field(
        description="Encouraging message to build confidence"
    )


class GuidedResponse(BaseModel):
    """
    Response from GUIDED mode - acknowledge effort and provide targeted hint.
    """
    
    acknowledgment: str = Field(
        description="What the student got right or understood partially"
    )
    
    missing_piece: str = Field(
        description="Explicitly state what's missing in their approach"
    )
    
    hint: str = Field(
        description="Clear hint pointing toward the correct path, tied to the missing piece"
    )
    
    encouragement: str = Field(
        description="Encouraging message"
    )


class ScaffoldResponse(BaseModel):
    """
    Response from SCAFFOLD mode.
    
    Handles BOTH modes in a single LLM call:
    - First instruction (retry_count == 0): Introduce step + check question
    - Evaluation (retry_count >= 1): Evaluate student's answer + respond
    """

    response_to_student: str = Field(
        description=(
            "Complete, warm, conversational message sent directly to the student. "
            "If presenting first instruction: introduce the step clearly and end with a check question. "
            "If evaluating: acknowledge their attempt, then either celebrate (if correct) or "
            "gently re-explain (if wrong). No markdown, no bullet points. 2-3 sentences max. "
            "Sound like a warm tutor sitting next to them, not a textbook."
        )
    )

    is_correct: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the student's answer to the check question was correct. "
            "Set to null/None when presenting the first instruction (no evaluation happening). "
            "Set to true/false when evaluating the student's response."
        )
    )

    should_advance: bool = Field(
        default=False,
        description=(
            "True if the student answered correctly and we should move to the next step. "
            "Must only be True when is_correct=True. "
            "False when presenting first instruction or when student answered incorrectly."
        )
    )


class ScaffoldRevealResponse(BaseModel):
    """
    Response when max scaffold retries are reached for a step.
    The LLM looks at the student's actual attempts in the conversation history,
    acknowledges them specifically, reveals the correct answer/method warmly,
    and encourages the student to move on.
    """

    acknowledgment: str = Field(
        description=(
            "A warm, specific acknowledgment of the student's attempts. "
            "Reference what they actually tried (from conversation history) — don't be generic. "
            "E.g., 'I can see you tried X — that was a good instinct!'"
        )
    )

    reveal: str = Field(
        description=(
            "Clear, kind reveal of the correct answer/method for this step. "
            "Include a brief one-sentence explanation of why it works. "
            "Use simple, age-appropriate language. Do NOT use markdown."
        )
    )

    encouragement: str = Field(
        description=(
            "Short encouragement transitioning to the next step. "
            "E.g., 'These things take practice — let's keep going!' "
            "Keep it warm and forward-looking."
        )
    )


class ConceptResponse(BaseModel):
    """
    Response from CONCEPT mode - teach missing prerequisite concept.
    """
    
    teaching_response: str = Field(
        description="A natural, conversational response that teaches the concept warmly. Include explanation with examples and end with a question to check understanding. Remember we want the text to be minimal.Speak like a friendly tutor having a conversation, not a textbook and continue on the previous conversaion and ensure you keep it to 2-3 sentences only."
    )


class ReflectionResponse(BaseModel):
    """
    Response from REFLECTION node - celebrate success and suggest next steps.
    """
    
    appreciation: str = Field(
        description="Warm appreciation of student's effort and success"
    )
    
    confidence_check: str = Field(
        description="Question asking how confident the student feels about this type of problem"
    )
    
    summary_offer: str = Field(
        description="A friendly question asking if they would like to review a step-by-step summary of the solution we just learned"
    )


class HandleSummaryResponse(BaseModel):
    """
    Response from HANDLE_SUMMARY_REQUEST node - interprets if student wanted summary and gives next steps.
    """
    wants_summary: bool = Field(
        description="Whether the student said they want to see the summary of steps"
    )
    
    response_prefix: str = Field(
        description="A short 1-sentence acknowledgment (e.g. 'Here you go!' or 'No problem!')"
    )
    
    next_action_suggestions: List[str] = Field(
        description="List of suggested next actions (e.g., 'Try a similar problem', 'Practice with different numbers')"
    )


class ConceptCheckResponse(BaseModel):
    """
    Response for checking if student knows required concepts.
    Used in initial ASSESSMENT to determine which concepts to teach.
    """
    
    missing_concepts: List[str] = Field(
        description="List of required concepts the student doesn't understand yet. Return empty list if all concepts are understood."
    )
    
    reasoning: str = Field(
        description="Brief explanation of which concepts are missing and why, based on student's response"
    )
    
    response_to_student: str = Field(
        description=(
            "A warm, encouraging message to send directly to the student. "
            "If concepts are missing: acknowledge their response, warmly mention you'll cover the missing concepts together before solving the problem (name them naturally). "
            "If no concepts are missing: praise their understanding and let them know you'll now look at their approach to the problem. "
            "Keep it brief, friendly, and age-appropriate (Class 7 student)."
        )
    )



class ApproachAssessmentResponse(BaseModel):
    """
    Response for assessing approach quality after concept teaching (or if no concepts missing).
    Scores Tu (understanding) and Ta (approach) to determine pedagogical mode.
    """
    
    Tu: float = Field(
        description="Understanding score (0-1). Criteria: identifies operation needed, understands problem terms/meaning, knows what result represents",
        ge=0.0,
        le=1.0
    )
    
    Ta: float = Field(
        description="Approach score (0-1). Criteria: mentions correct method, logical step order, handles conversion/edge cases",
        ge=0.0,
        le=1.0
    )
    
    reasoning: str = Field(
        description="Brief explanation of the scores and what the student understands vs. struggles with"
    )
    
    @field_validator('Tu', 'Ta')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {v}")
        return v


class ConceptEvaluationResponse(BaseModel):
    """
    Response when evaluating student's micro-check answer during concept teaching.
    LLM evaluates student understanding AND generates response in a single call.
    Used in try-counter pattern (max 3 tries).
    """
    
    understood: bool = Field(
        description="Whether the student's response shows understanding of the concept"
    )
    
    next_state: Literal["move_on", "stay"] = Field(
        description="'move_on' if concept is understood or max tries reached. 'stay' if need to re-teach."
    )
    
    response_to_student: str = Field(
        description="Message to student. Either: (1) praise + confirmation if understood(Remember do NOT mention the original problem in any way whatsoever.Also do not say anything about moving on), OR (2) re-explanation + micro-check question again if not understood.Ensure that your response is not more than 2-3 sentences."
    )


class StartAnswerCheckResponse(BaseModel):
    """
    Structured response for evaluating the student's answer at the very start.
    The LLM is given the correct answer in its context ONLY — it must NEVER
    reveal it to the student in feedback.
    """

    is_correct: bool = Field(
        description="Whether the student's answer matches the correct answer."
    )

    feedback: str = Field(
        description=(
            "Short message (2-3 sentences max) sent directly to the student. "
            "If correct: warm congratulations then ask 'Would you like me to walk through the steps, or shall we move on?'. "
            "If wrong attempt 1: begin with 'That is incorrect.' only if a numerical answer is given. Give ONE brief conceptual hint to help them find the final answer. CRITICAL: Do NOT ask any intermediate questions or ask them to calculate partial steps. Simply provide the hint and encourage them to try finding the FULL and FINAL answer again. Do NOT reveal the answer. "
            "If wrong attempt 2: begin with 'That is incorrect.' only if a numerical answer is given. Say you will work through it together. Do NOT reveal the answer, and do not give any hints. Just say we will solve it together. "
            "NEVER include the correct answer value anywhere in this field."
        )
    )
