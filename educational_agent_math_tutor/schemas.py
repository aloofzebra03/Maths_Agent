"""
Pydantic schemas and state definition for the Math Tutoring Agent.
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# Type alias for pedagogical modes
Mode = Literal["coach", "guided", "scaffold", "concept"]


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
    
    # Step tracking
    steps: List[Dict[str, Any]]  # Full step objects with step_id, description, concept
    step_index: int
    max_steps: int
    current_step_description: Optional[str]
    
    # Pedagogical context
    missing_concept: Optional[str]
    previous_mode: Optional[Mode]
    nudge_count: int
    scaffold_retry_count: int
    
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
    Response from SCAFFOLD mode - provide one concrete operation for current step.
    """
    
    step_instruction: str = Field(
        description="Clear, concrete instruction for exactly one operation the student should perform"
    )
    
    step_context: str = Field(
        description="Brief context of why this step is needed (age-appropriate)"
    )
    
    check_question: Optional[str] = Field(
        default=None,
        description="Simple question to verify the student completed this step correctly"
    )


class ConceptResponse(BaseModel):
    """
    Response from CONCEPT mode - teach missing prerequisite concept.
    """
    
    concept_explanation: str = Field(
        description="Clear explanation of the concept using Class 7 appropriate language and analogy"
    )
    
    analogy: str = Field(
        description="Relatable analogy or visual description to make the concept concrete"
    )
    
    micro_check_question: str = Field(
        description="One simple question to verify understanding of this concept before resuming"
    )
    
    encouragement: str = Field(
        description="Encouraging message to build confidence"
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
    
    next_action_suggestions: List[str] = Field(
        description="List of suggested next actions (e.g., 'Try a similar problem', 'Practice with different numbers', 'Learn a new concept')"
    )
