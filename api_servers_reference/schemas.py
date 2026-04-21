from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class SessionMetadata(BaseModel):

    # Simulation flags
    show_simulation: bool = Field(
        default=False,
        description="Whether a simulation should be displayed to the student. When true, check simulation_config for parameters."
    )
    simulation_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for the simulation (type, parameters, etc.). Empty dict if no simulation is active."
    )
    
    # Image metadata
    image_url: Optional[str] = Field(
        default=None,
        description="Base64-encoded image URL (e.g., 'data:image/png;base64,...'). None if no image is present."
    )
    image_description: Optional[str] = Field(
        default=None,
        description="Description of the image. None if no image is present."
    )
    image_node: Optional[str] = Field(
        default=None,
        description="The pedagogical node where the image was generated (e.g., 'CI', 'GE', 'APK'). None if no image."
    )

    video_url: Optional[str] = Field(
        default=None,
        description="Base64-encoded video URL (e.g., 'data:video/mp4;base64,...'). None if no video is present."
    )
    video_node: Optional[str] = Field(
        default=None,
        description="The pedagogical node where the video was generated (e.g., 'CI', 'GE', 'APK'). None if no video."
    )
    
    # Scores and progress
    quiz_score: float = Field(
        default=-1.0,
        description="Student's quiz performance score from 0.0 to 1.0. Set to -1.0 if no quiz has been taken yet."
    )
    retrieval_score: float = Field(
        default=-1.0,
        description="RAG retrieval confidence score from 0.0 to 1.0. Set to -1.0 if no retrieval has occurred."
    )
    
    # Concept tracking
    sim_concepts: List[str] = Field(
        default_factory=list,
        description="List of concepts in the simulation learning sequence. Empty list if not in simulation mode."
    )
    sim_current_idx: int = Field(
        default=-1,
        description="Index of the current concept being taught in simulation (0-based). Set to -1 if not in simulation."
    )
    sim_total_concepts: int = Field(
        default=0,
        description="Total number of concepts in the simulation sequence. Set to 0 if not in simulation mode."
    )
    
    # Misconception tracking
    misconception_detected: bool = Field(
        default=False,
        description="Whether a misconception was detected in the student's latest response."
    )
    last_correction: str = Field(
        default="",
        description="The correction message provided for the most recent misconception. Empty string if no misconception."
    )
    
    # Node transitions
    node_transitions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of pedagogical node transitions (e.g., [{from: 'APK', to: 'CI', timestamp: '...'}]). Empty list at session start."
    )


class StartSessionResponse(BaseModel):
    success: bool = Field(
        description="Whether the session was started successfully"
    )
    session_id: str = Field(
        description="Unique session identifier for tracking purposes"
    )
    thread_id: str = Field(
        description="Unique thread ID for continuing this session (store this on client side)"
    )
    problem_id: Optional[str] = Field(
        default=None,
        description="Resolved problem identifier used by the agent."
    )
    user_id: str = Field(
        description="User identifier ('anonymous' if not provided in request)"
    )
    agent_response: str = Field(
        description="The agent's initial greeting or first teaching message"
    )
    current_state: str = Field(
        description="Current pedagogical node/state (e.g., 'APK', 'CI', 'GE', 'AR', 'TC', 'RLC', 'END')"
    )
    message: str = Field(
        default="Session started successfully",
        description="Status message about the session creation"
    )
    metadata: SessionMetadata = Field(
        default_factory=SessionMetadata,
        description="Session metadata with scores, images, simulation status, etc."
    )


class ContinueSessionResponse(BaseModel):
    success: bool = Field(
        description="Whether the agent response was generated successfully"
    )
    thread_id: str = Field(
        description="The thread ID of this session (same as request)"
    )
    agent_response: str = Field(
        description="The agent's response to the student's message (teaching, questions, feedback, etc.)"
    )
    current_state: str = Field(
        description="Current pedagogical node/state after processing the student's message"
    )
    metadata: SessionMetadata = Field(
        default_factory=SessionMetadata,
        description="Session metadata with scores, images, simulation status, misconceptions, etc."
    )
    message: str = Field(
        default="Response generated successfully",
        description="Status message about the response generation"
    )


class StartSessionRequest(BaseModel):
    """Minimal request model for starting a learning session."""

    problem_id: Optional[str] = Field(
        default=None,
        description="Preferred problem identifier from /problems (e.g., 'add_frac_diff_den_01')."
    )
    student_id: Optional[str] = Field(
        default=None,
        description="Optional unique student identifier."
    )
    session_label: Optional[str] = Field(
        default=None,
        description="Optional session label used to build thread/session IDs."
    )
    is_kannada: bool = Field(
        default=False,
        description="Set true for Kannada mode, false for English mode."
    )


class ContinueSessionRequest(BaseModel):
    """Minimal request model for continuing a session."""

    thread_id: str = Field(
        ...,
        description="Thread ID returned by /session/start."
    )
    user_message: str = Field(
        ...,
        description="Student message text (or image path when multipart upload is used)."
    )


class ProblemInfo(BaseModel):
    problem_id: str = Field(description="Unique problem identifier")
    topic: str = Field(description="Human-readable topic/title for the problem")
    difficulty: Optional[str] = Field(default=None, description="Difficulty label if present")


class ProblemsListResponse(BaseModel):
    success: bool = Field(default=True, description="Whether the problems were retrieved successfully")
    problems: List[ProblemInfo] = Field(description="List of available problems")
    total: int = Field(description="Total number of available problems")
    message: str = Field(default="Available problems retrieved successfully", description="Status message")
