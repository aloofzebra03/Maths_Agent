from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


# ============================================================================
# REQUEST MODELS
# ============================================================================

class StartSessionRequest(BaseModel):
    """Request to start a new learning session with the educational agent."""
    concept_title: str = Field(
        ..., 
        description="The educational concept to teach (e.g., 'Pendulum and its Time Period', 'Photosynthesis')"
    )
    student_id: Optional[str] = Field(
        None, 
        description="Optional unique identifier for the student (for tracking and analytics)"
    )
    persona_name: Optional[str] = Field(
        None, 
        description="Optional test persona name for simulated student behavior (e.g., 'Confused Student', 'Eager Student')"
    )
    session_label: Optional[str] = Field(
        None, 
        description="Optional custom label for this session (used in thread_id generation)"
    )
    is_kannada: bool = Field(
        default=False,
        description="Whether to conduct the session in Kannada language. Default is False (English)."
    )
    # model: Optional[str] = Field(
    #     default="gemma-3-27b-it",
    #     description="Gemini model to use for this session. Available: gemma-3-27b-it, gemma-3-27b-it-exp, gemini-1.5-flash, gemini-1.5-pro. Defaults to gemma-3-27b-it."
    # )
    student_level: str = Field(
        default="medium",
        description="Student ability level for dynamic autosuggestions. Options: 'low', 'medium', 'advanced'. Defaults to 'medium'."
    )


class ContinueSessionRequest(BaseModel):
    """Request to continue an existing learning session with a student's message."""
    thread_id: str = Field(
        ..., 
        description="The unique thread ID of the session to continue (returned from /session/start)"
    )
    user_message: str = Field(
        ..., 
        description="The student's message or response to the agent's previous question"
    )
    # model: Optional[str] = Field(
    #     default="gemma-3-27b-it",
    #     description="Optional: Override the model for this specific request. If not provided, uses the model from session start."
    # )
    clicked_autosuggestion: Optional[bool] = Field(
        default=False,
        description="True if user clicked an autosuggestion button, False if typed message"
    )
    student_level: Optional[str] = Field(
        default=None,
        description="Optional: Update student ability level mid-session. Options: 'low', 'medium', 'advanced'."
    )


class SessionStatusRequest(BaseModel):
    """Request to get the current status of a learning session."""
    thread_id: str = Field(
        ..., 
        description="The unique thread ID of the session to check"
    )


class TestPersonaRequest(BaseModel):
    """Request to start a test session with a predefined student persona."""
    persona_name: str = Field(
        ..., 
        description="Name of the test persona. Available: 'Eager Student', 'Confused Student', 'Distracted Student', 'Dull Student'"
    )
    concept_title: str = Field(
        default="Pendulum and its Time Period",
        description="The educational concept to teach to the test persona"
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================

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
    user_id: str = Field(
        description="User identifier ('anonymous' if not provided in request)"
    )
    agent_response: str = Field(
        description="The agent's initial greeting or first teaching message"
    )
    current_state: str = Field(
        description="Current pedagogical node/state (e.g., 'APK', 'CI', 'GE', 'AR', 'TC', 'RLC', 'END')"
    )
    concept_title: str = Field(
        description="The concept being taught in this session"
    )
    message: str = Field(
        default="Session started successfully",
        description="Status message about the session creation"
    )
    metadata: SessionMetadata = Field(
        default_factory=SessionMetadata,
        description="Session metadata with scores, images, simulation status, etc."
    )
    autosuggestions: List[str] = Field(
        default_factory=list,
        description="Contextually appropriate quick-reply suggestions for the user (translated if Kannada)"
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
    autosuggestions: List[str] = Field(
        default_factory=list,
        description="Contextually appropriate quick-reply suggestions for the user (translated if Kannada)"
    )


class SessionStatusResponse(BaseModel):
    success: bool = Field(
        description="Whether the status was retrieved successfully"
    )
    thread_id: str = Field(
        description="The thread ID of the session"
    )
    exists: bool = Field(
        description="Whether the session exists in the checkpoint store"
    )
    current_state: Optional[str] = Field(
        None,
        description="Current pedagogical node if session exists (e.g., 'APK', 'CI', 'GE')"
    )
    progress: Optional[Dict[str, Any]] = Field(
        None,
        description="Progress information: nodes visited (asked_apk, asked_ci, etc.), concepts, simulation status"
    )
    concept_title: Optional[str] = Field(
        None,
        description="The concept being taught in this session"
    )
    message: str = Field(
        default="Status retrieved successfully",
        description="Status message about the retrieval"
    )


class SessionHistoryResponse(BaseModel):
    success: bool = Field(
        description="Whether the history was retrieved successfully"
    )
    thread_id: str = Field(
        description="The thread ID of the session"
    )
    exists: bool = Field(
        description="Whether the session exists in the checkpoint store"
    )
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of conversation messages with role ('user' or 'assistant'), content, and node information"
    )
    node_transitions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of pedagogical node transitions during the session"
    )
    concept_title: Optional[str] = Field(
        None,
        description="The concept being taught in this session"
    )
    message: str = Field(
        default="History retrieved successfully",
        description="Status message about the retrieval"
    )


class SessionSummaryResponse(BaseModel):
    success: bool = Field(
        description="Whether the summary was retrieved successfully"
    )
    thread_id: str = Field(
        description="The thread ID of the session"
    )
    exists: bool = Field(
        description="Whether the session exists in the checkpoint store"
    )
    summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed session summary with learning outcomes and metrics"
    )
    quiz_score: Optional[float] = Field(
        None,
        description="Final quiz score from 0.0 to 1.0 (null if no quiz taken)"
    )
    transfer_success: Optional[bool] = Field(
        None,
        description="Whether knowledge transfer was successful (null if not assessed)"
    )
    misconception_detected: Optional[bool] = Field(
        None,
        description="Whether any misconceptions were detected during the session"
    )
    definition_echoed: Optional[bool] = Field(
        None,
        description="Whether the student successfully echoed/restated the definition"
    )
    message: str = Field(
        default="Summary retrieved successfully",
        description="Status message about the retrieval"
    )


class HealthResponse(BaseModel):
    status: str = Field(
        description="Health status of the API (e.g., 'healthy')"
    )
    version: str = Field(
        description="API version number"
    )
    persistence: str = Field(
        description="Type of persistence/checkpoint storage being used (e.g., 'InMemorySaver', 'PostgreSQL')"
    )
    agent_type: str = Field(
        description="Type of educational agent being used"
    )
    available_endpoints: List[str] = Field(
        description="List of all available API endpoints"
    )


class ErrorResponse(BaseModel):
    success: bool = Field(
        default=False,
        description="Always false for error responses"
    )
    error: str = Field(
        description="Short error type or category"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error message explaining what went wrong"
    )


class PersonaInfo(BaseModel):
    name: str = Field(
        description="Name of the persona (e.g., 'Eager Student', 'Confused Student')"
    )
    description: str = Field(
        description="Description of the persona's behavior and characteristics"
    )
    sample_phrases: List[str] = Field(
        description="Example phrases this persona might use when responding to the agent"
    )


class PersonasListResponse(BaseModel):
    success: bool = Field(
        default=True,
        description="Whether the personas were retrieved successfully"
    )
    personas: List[PersonaInfo] = Field(
        description="List of available test personas with their details"
    )
    total: int = Field(
        description="Total number of available personas"
    )
    message: str = Field(
        default="Available test personas retrieved successfully",
        description="Status message about the retrieval"
    )


class TestImageRequest(BaseModel):
    concept_title: str = Field(
        ...,
        description="The concept to get an image for (e.g., 'Pendulum and its Time Period')"
    )
    definition_context: str = Field(
        default="",
        description="Optional definition/explanation context to help select the most relevant image"
    )


class TestImageResponse(BaseModel):
    success: bool = Field(
        description="Whether the image was retrieved successfully"
    )
    concept: str = Field(
        description="The concept the image is for"
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Base64-encoded image URL. None if no image found."
    )
    image_description: Optional[str] = Field(
        default=None,
        description="Description of the image"
    )
    message: str = Field(
        description="Status message about the retrieval"
    )


class TestSimulationRequest(BaseModel):
    concept_title: str = Field(
        ...,
        description="The concept to get simulation config for (e.g., 'Pendulum and its Time Period')"
    )
    simulation_type: Optional[str] = Field(
        default=None,
        description="Optional specific simulation type (e.g., independent variable to vary like 'length', 'gravity', etc.)"
    )


class TestSimulationResponse(BaseModel):
    success: bool = Field(
        description="Whether the simulation config was retrieved successfully"
    )
    concept: str = Field(
        description="The concept the simulation is for"
    )
    simulation_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Simulation configuration with type, parameters, agent message, etc."
    )
    message: str = Field(
        description="Status message about the retrieval"
    )


class ConceptsListResponse(BaseModel):
    success: bool = Field(
        default=True,
        description="Whether the concepts were retrieved successfully"
    )
    concepts: List[str] = Field(
        description="List of all available concepts (title case)"
    )
    total: int = Field(
        description="Total number of available concepts"
    )
    message: str = Field(
        default="Available concepts retrieved successfully",
        description="Status message about the retrieval"
    )


# ============================================================================
# CONCEPT MAP SCHEMAS (from concept_map_poc integration)
# ============================================================================

class ConceptMapRequest(BaseModel):
    """Request model for concept map timeline generation."""
    description: str = Field(
        ...,
        min_length=1,
        description="Educational description text to analyze (1 word to 3000+ words)",
        example="Photosynthesis is the process by which plants convert light energy into chemical energy."
    )
    educational_level: str = Field(
        default="high school",
        description="Target educational level for concept extraction",
        example="high school"
    )
    topic_name: Optional[str] = Field(
        default=None,
        description="Optional topic name (auto-extracted if not provided)",
        example="Photosynthesis"
    )

    class Config:
        schema_extra = {
            "example": {
                "description": "Photosynthesis is the process by which green plants convert light energy into chemical energy using chlorophyll in chloroplasts.",
                "educational_level": "high school",
                "topic_name": "Photosynthesis"
            }
        }


class ConceptMapResponse(BaseModel):
    """Response model for concept map timeline generation."""
    success: bool = Field(
        description="Whether the operation was successful"
    )
    filepath: str = Field(
        description="Path to the saved JSON file in concept_json_timings/ folder"
    )
    timeline: Dict[str, Any] = Field(
        description="Complete timeline data with concepts and reveal times"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "filepath": "concept_json_timings/photosynthesis_20251221_143022.json",
                "timeline": {
                    "metadata": {
                        "topic_name": "Photosynthesis",
                        "educational_level": "high school",
                        "total_duration": 25.5,
                        "total_concepts": 6
                    },
                    "concepts": [
                        {
                            "name": "Photosynthesis",
                            "reveal_time": 0.0,
                            "importance_rank": 1
                        }
                    ]
                }
            }
        }
