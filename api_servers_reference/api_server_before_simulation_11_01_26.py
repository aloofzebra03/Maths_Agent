from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, Optional, Any
import sys
import os
import traceback
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import educational agent
# sys.path.insert(0, str(Path(__file__).parent.parent))

from educational_agent_optimized_langsmith_autosuggestion.graph import graph
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from api_servers_reference.schemas import (
    StartSessionRequest, StartSessionResponse,
    ContinueSessionRequest, ContinueSessionResponse,
    SessionStatusRequest, SessionStatusResponse,
    SessionHistoryResponse, SessionSummaryResponse,
    TestPersonaRequest, HealthResponse, ErrorResponse,
    PersonaInfo, PersonasListResponse,
    TestImageRequest, TestImageResponse,
    TestSimulationRequest, TestSimulationResponse,
    ConceptsListResponse,
    ConceptMapRequest, ConceptMapResponse
)

# Import personas from tester_agent
from tester_agent.personas import personas

# Import utility functions for testing
from utils.shared_utils import (
    select_most_relevant_image_for_concept_introduction,
    create_simulation_config,
    get_all_available_concepts,
)

from api_tracker_utils.error import MinuteLimitExhaustedError, DayLimitExhaustedError
from api_tracker_utils.tracker import track_model_call

# Import concept map functions from external repo
from concept_map_poc.timeline_mapper import create_timeline
from concept_map_poc.streamlit_app_standalone import save_timeline_json_to_disk

# Import for wrapper function
from contextlib import contextmanager

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Educational Agent API",
    version="1.0.0",
    description="Stateful API for personalized education with LangGraph-based agent"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@contextmanager
def use_google_api_key():
    """
    Context manager to temporarily set GOOGLE_API_KEY from GOOGLE_API_KEY_1
    for concept map functions that expect GOOGLE_API_KEY.
    
    This allows integration with external concept_map_poc code without modifying it.
    """
    import google.generativeai as genai
    
    original = os.environ.get('GOOGLE_API_KEY')
    try:
        # Set GOOGLE_API_KEY to your key for the duration of the context
        api_key = os.getenv('GOOGLE_API_KEY_1')
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key
            # Reconfigure genai with the new API key
            genai.configure(api_key=api_key)
            print(f"🔑 Configured Google Generative AI with GOOGLE_API_KEY_1")
        yield
    finally:
        # Restore original value after context exits
        if original:
            os.environ['GOOGLE_API_KEY'] = original
            genai.configure(api_key=original)
        else:
            os.environ.pop('GOOGLE_API_KEY', None)


def generate_thread_id(concept_title: str, is_kannada: bool = False, label: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """
    Generate a unique thread ID with ordered components for better organization.
    
    Args:
        concept_title: The concept being taught
        is_kannada: Whether the session is in Kannada
        label: Optional custom session label
        user_id: Optional user/student ID
    
    Returns:
        Formatted thread ID: <user_id>-<label>-<concept>-<lang>-thread-<timestamp>
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Build parts in order: user_id, label, concept, language
    parts = []
    
    # 1. User ID (if present)
    if user_id:
        parts.append(user_id)
    
    # 2. Label (if present)
    if label:
        parts.append(label)
    
    # 3. Concept (always present, cleaned)
    concept_slug = concept_title.lower().replace(" ", "-").replace("'", "").replace(",", "")
    parts.append(concept_slug)
    
    # 4. Language indicator
    lang = "kannada" if is_kannada else "english"
    parts.append(lang)
    
    # If no user_id or label, add "session" at the beginning
    if not user_id and not label:
        parts.insert(0, "session")
    
    # Join all parts with thread indicator and timestamp
    return f"{'-'.join(parts)}-thread-{timestamp}"


def validate_student_level(level: str) -> str:
    """
    Validate student level, default to medium if invalid.
    
    Args:
        level: Student level string
    
    Returns:
        Validated level: "low", "medium", or "advanced"
    """
    valid_levels = ["low", "medium", "advanced"]
    if level not in valid_levels:
        print(f"⚠️ Invalid student_level '{level}', defaulting to 'medium'")
        return "medium"
    return level


def get_state_from_checkpoint(thread_id: str) -> Optional[Dict[str, Any]]:
    try:
        # Get the state snapshot from the graph using the thread_id
        state_snapshot = graph.get_state(config={"configurable": {"thread_id": thread_id}})
        
        # Check if state exists and has values
        if state_snapshot and state_snapshot.values:
            return state_snapshot.values
        return None
    except Exception as e:
        print(f"Error retrieving state for thread {thread_id}: {e}")
        return None


def extract_metadata_from_state(state: Dict[str, Any]):
    from api_servers_reference.schemas import SessionMetadata
    
    # Extract image metadata (only image URL and node)
    image_url = None
    image_description = None
    image_node = None
    video_url = None
    video_node = None
    
    enhanced_meta = state.get("enhanced_message_metadata")
    if enhanced_meta:
        image_url = enhanced_meta.get("image")
        image_description = image_url.get("description")
        image_url = image_url.get("url")
        image_node = enhanced_meta.get("node")
        video_url = enhanced_meta.get("video")
        video_node = enhanced_meta.get("video_node")
    
    # Build metadata with consistent structure - all fields present with defaults
    return SessionMetadata(
        # Simulation flags
        show_simulation=state.get("show_simulation", False),
        simulation_config=state.get("simulation_config", {}) if state.get("show_simulation") else {},
        
        # Image metadata
        image_url=image_url,
        image_description=image_description,
        image_node=image_node,

        # Video metadata
        video_url=video_url,
        video_node=video_node,
        
        # Scores and progress (-1.0 means not set yet)
        quiz_score=state.get("quiz_score", -1.0) if state.get("quiz_score") is not None else -1.0,
        retrieval_score=state.get("retrieval_score", -1.0) if state.get("retrieval_score") is not None else -1.0,
        
        # Concept tracking
        sim_concepts=state.get("sim_concepts", []),
        sim_current_idx=state.get("sim_current_idx", -1),
        sim_total_concepts=state.get("sim_total_concepts", 0),
        
        # Misconception tracking
        misconception_detected=state.get("misconception_detected", False),
        last_correction=state.get("last_correction", ""),
        
        # Node transitions
        node_transitions=state.get("node_transitions", [])
    )


def get_history_from_state(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    history = []
    messages = state.get("messages", [])
    
    for msg in messages:
        if hasattr(msg, 'type'):
            if msg.type == "human":
                # Skip the initial "__start__" message
                if msg.content != "__start__":
                    history.append({
                        "role": "user",
                        "content": msg.content
                    })
            elif msg.type == "ai":
                current_node = state.get("current_state", "unknown")
                history.append({
                    "role": "assistant",
                    "content": msg.content,
                    "node": current_node
                })
    
    return history


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, Any])
def read_root():
    return {
        "message": "Educational Agent API is running!",
        "version": "1.0.0",
        "agent_type": "educational_agent_optimized_langsmith",
        "endpoints": [
            "GET  /health - Health check",
            "GET  /concepts - List all available concepts",
            "POST /session/start - Start new learning session",
            "POST /session/continue - Continue existing session",
            "GET  /session/status/{thread_id} - Get session status",
            "GET  /session/history/{thread_id} - Get conversation history",
            "GET  /session/summary/{thread_id} - Get session summary",
            "DELETE /session/{thread_id} - Delete session",
            "GET  /test/personas - List available test personas",
            "POST /test/persona - Test with predefined persona",
            "POST /test/images - Get image for a concept",
            "POST /test/simulation - Get simulation config for a concept",
            "POST /concept-map/generate - Generate concept map timeline from description"
        ]
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        persistence="Postgres (Supabase))",
        agent_type="educational_agent_optimized_langsmith_autosuggestion",
        available_endpoints=[
            "/",
            "/health",
            "/concepts",
            "/session/start",
            "/session/continue",
            "/session/status/{thread_id}",
            "/session/history/{thread_id}",
            "/session/summary/{thread_id}",
            "/session/{thread_id}",
            "/test/personas",
            "/test/persona",
            "/test/images",
            "/test/simulation",
            "/concept-map/generate"
        ]
    )


@app.get("/concepts", response_model=ConceptsListResponse)
def list_available_concepts():
    """List all available concepts that can be taught by the educational agent."""
    try:
        print("API /concepts - Retrieving all available concepts")
        
        concepts = get_all_available_concepts()
        
        # Convert concepts to proper title case for display
        concepts_title_case = [' '.join(word.capitalize() for word in concept.split()) for concept in concepts]
        
        return ConceptsListResponse(
            success=True,
            concepts=concepts_title_case,
            total=len(concepts_title_case),
            message=f"Retrieved {len(concepts_title_case)} available concepts"
        )
        
    except Exception as e:
        print(f"API error in /concepts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving concepts: {str(e)}")


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(request: StartSessionRequest):
    try:
        print(f"API /session/start - concept: {request.concept_title}, student: {request.student_id}, language: {'Kannada' if request.is_kannada else 'English'}")
        
        # Generate unique thread_id with concept and language info
        thread_id = generate_thread_id(
            concept_title=request.concept_title,
            is_kannada=request.is_kannada,
            label=request.session_label,
            user_id=request.student_id
        )
        
        # Generate session_id and user_id
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = request.session_label or request.persona_name or "session"
        session_id = f"{base}-{timestamp}"
        user_id = request.student_id or "anonymous"
        
        print(f"📌 Generated thread_id: {thread_id}")
        
        # # Validate model if provided
        # model = request.model or "gemma-3-27b-it"
        # if model not in AVAILABLE_GEMINI_MODELS:
        #     raise HTTPException(
        #         status_code=400,
        #         detail=f"Invalid model '{model}'. Available models: {AVAILABLE_GEMINI_MODELS}"
        #     )
        
        # Validate student level
        student_level = validate_student_level(request.student_level)
        print(f"📊 Student level: {student_level}")
        
        # Start the conversation by invoking the graph with __start__ message
        # Tracker will automatically select the best API key and model based on rate limits
        print("Invoking graph to start session (tracker will select optimal model)")
        result = graph.invoke(
            {
                "messages": [HumanMessage(content="__start__")],
                "is_kannada": request.is_kannada,
                "concept_title": request.concept_title,
                "student_level": student_level,
                "summary": "",  # Initialize summary field
                "summary_last_index": -1,  # Initialize summary tracking
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        
        # Extract agent response
        agent_response = result.get("agent_output", "")
        if not agent_response and result.get("messages"):
            # Fallback: get last AI message
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == "ai":
                    agent_response = msg.content
                    break
        
        # Extract metadata
        metadata = extract_metadata_from_state(result)
        
        # Extract autosuggestions
        autosuggestions = result.get("autosuggestions", [])
        
        return StartSessionResponse(
            success=True,
            session_id=session_id,
            thread_id=thread_id,
            user_id=user_id,
            agent_response=agent_response,
            current_state=result.get("current_state", "START"),
            concept_title=request.concept_title,
            message="Session started successfully. Agent is ready for student input.",
            metadata=metadata,
            autosuggestions=autosuggestions
        )

    except MinuteLimitExhaustedError as e:
        print(f"[API] Error processing query: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code = 501,
            detail=f"Error processing query: {e}"
        ) 
    
    except DayLimitExhaustedError as e:
        print(f"[API] Error processing query: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code = 502,
            detail=f"Error processing query: {e}"
        )
    
    except Exception as e:
        print(f"API error in /session/start: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")


@app.post("/session/continue", response_model=ContinueSessionResponse)
def continue_session(request: ContinueSessionRequest):
    try:
        print(f"API /session/continue - thread: {request.thread_id}, message: {request.user_message[:50]}...")
        
        # Check if session exists by trying to get its state
        existing_state = get_state_from_checkpoint(request.thread_id)
        if existing_state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found for thread_id: {request.thread_id}. Please start a new session."
            )
        
        # Validate model if provided
        update_dict = {
            "messages": [HumanMessage(content=request.user_message)],
            # "last_user_msg": request.user_message,
            "clicked_autosuggestion": request.clicked_autosuggestion
        }
        
        # Allow updating student level mid-session
        if request.student_level:
            validated_level = validate_student_level(request.student_level)
            update_dict["student_level"] = validated_level
            print(f"📊 Updated student level to: {validated_level}")
        
        # Continue the conversation using Command (resume)
        cmd = Command(
            resume=True,
            update=update_dict,
        )
        
        # Invoke graph with the user message
        result = graph.invoke(
            cmd,
            config={"configurable": {"thread_id": request.thread_id}},
        )
        
        # Debug: Print result keys to understand structure
        print(f"🔍 DEBUG - Result keys: {list(result.keys())}")
        print(f"🔍 DEBUG - current_state value: {result.get('current_state')}")
        print(f"🔍 DEBUG - current_state type: {type(result.get('current_state'))}")
        
        # Extract agent response
        agent_response = result.get("agent_output", "")
        if not agent_response and result.get("messages"):
            # Fallback: get last AI message
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == "ai":
                    agent_response = msg.content
                    break
        
        # Extract metadata
        metadata = extract_metadata_from_state(result)
        
        # Extract autosuggestions
        autosuggestions = result.get("autosuggestions", [])
        
        return ContinueSessionResponse(
            success=True,
            thread_id=request.thread_id,
            agent_response=agent_response,
            current_state=result.get("current_state", "UNKNOWN"),
            metadata=metadata,
            message="Response generated successfully",
            autosuggestions=autosuggestions
        )
    
    except MinuteLimitExhaustedError as e:
        print(f"[API] Error processing query: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code = 501,
            detail=f"Error processing query: {e}"
        ) 
    
    except DayLimitExhaustedError as e:
        print(f"[API] Error processing query: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code = 502,
            detail=f"Error processing query: {e}"
        )
        
    except HTTPException:
        raise

    except Exception as e:
        print(f"API error in /session/continue: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error continuing session: {str(e)}")


@app.get("/session/status/{thread_id}", response_model=SessionStatusResponse)
def get_session_status(thread_id: str):
    try:
        print(f"API /session/status - thread: {thread_id}")
        
        # Get state from checkpoint
        state = get_state_from_checkpoint(thread_id)
        if state is None:
            return SessionStatusResponse(
                success=True,
                thread_id=thread_id,
                exists=False,
                message="Session not found"
            )
        
        progress = {
            "current_state": state.get("current_state", "UNKNOWN"),
            "asked_apk": state.get("asked_apk", False),
            "asked_ci": state.get("asked_ci", False),
            "asked_ge": state.get("asked_ge", False),
            "asked_ar": state.get("asked_ar", False),
            "asked_tc": state.get("asked_tc", False),
            "asked_rlc": state.get("asked_rlc", False),
            "concepts": state.get("sim_concepts", []),
            "current_concept_idx": state.get("sim_current_idx", 0),
            "total_concepts": len(state.get("sim_concepts", [])),
            "in_simulation": state.get("in_simulation", False),
            "misconception_detected": state.get("misconception_detected", False),
        }
        
        return SessionStatusResponse(
            success=True,
            thread_id=thread_id,
            exists=True,
            current_state=state.get("current_state", "UNKNOWN"),
            progress=progress,
            concept_title=state.get("concept_title", "Unknown Concept"),
            message="Status retrieved successfully"
        )
        
    except Exception as e:
        print(f"API error in /session/status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")


@app.get("/session/history/{thread_id}", response_model=SessionHistoryResponse)
def get_session_history(thread_id: str):
    try:
        print(f"API /session/history - thread: {thread_id}")
        
        # Get state from checkpoint
        state = get_state_from_checkpoint(thread_id)
        if state is None:
            return SessionHistoryResponse(
                success=True,
                thread_id=thread_id,
                exists=False,
                message="Session not found"
            )
        
        # Get conversation history
        history = get_history_from_state(state)
        
        # Get node transitions
        node_transitions = state.get("node_transitions", [])
        
        return SessionHistoryResponse(
            success=True,
            thread_id=thread_id,
            exists=True,
            messages=history,
            node_transitions=node_transitions,
            concept_title=state.get("concept_title", "Unknown Concept"),
            message="History retrieved successfully"
        )
        
    except Exception as e:
        print(f"API error in /session/history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@app.get("/session/summary/{thread_id}", response_model=SessionSummaryResponse)
def get_session_summary(thread_id: str):
    try:
        print(f"API /session/summary - thread: {thread_id}")
        
        # Get state from checkpoint
        state = get_state_from_checkpoint(thread_id)
        if state is None:
            return SessionSummaryResponse(
                success=True,
                thread_id=thread_id,
                exists=False,
                message="Session not found"
            )
        
        summary = state.get("session_summary", {})
        
        return SessionSummaryResponse(
            success=True,
            thread_id=thread_id,
            exists=True,
            summary=summary,
            quiz_score=state.get("quiz_score"),
            transfer_success=state.get("transfer_success"),
            misconception_detected=state.get("misconception_detected"),
            definition_echoed=state.get("definition_echoed"),
            message="Summary retrieved successfully"
        )
        
    except Exception as e:
        print(f"API error in /session/summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving summary: {str(e)}")


@app.delete("/session/{thread_id}")
def delete_session(thread_id: str):
    try:
        print(f"API /session DELETE - thread: {thread_id}")
        
        # Check if session exists
        state = get_state_from_checkpoint(thread_id)
        if state is None:
            return {
                "success": False,
                "thread_id": thread_id,
                "message": "Session not found"
            }
        
        try:
            from educational_agent_optimized_langsmith_autosuggestion.graph import checkpointer
            
            # Get the connection pool from the checkpointer
            if hasattr(checkpointer, 'conn'):
                # For ConnectionPool, we need to get a connection first
                with checkpointer.conn.connection() as conn:
                    with conn.cursor() as cur:
                        # Delete from checkpoints table where thread_id matches
                        cur.execute(
                            "DELETE FROM checkpoints WHERE thread_id = %s",
                            (thread_id,)
                        )
                        deleted_checkpoints = cur.rowcount
                        
                        # Delete from checkpoint_writes table where thread_id matches
                        cur.execute(
                            "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                            (thread_id,)
                        )
                        deleted_writes = cur.rowcount
                        
                        # Delete from checkpoint_blobs table if it exists
                        try:
                            cur.execute(
                                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                                (thread_id,)
                            )
                            deleted_blobs = cur.rowcount
                        except:
                            deleted_blobs = 0
                        
                        # Commit the transaction (required for autocommit=True in pool)
                        conn.commit()
                        
                        print(f"🗑️ Deleted {deleted_checkpoints} checkpoints, {deleted_writes} writes, {deleted_blobs} blobs for thread {thread_id}")
                        
                        return {
                            "success": True,
                            "thread_id": thread_id,
                            "message": f"Session deleted successfully from Postgres (removed {deleted_checkpoints} checkpoint records)"
                        }
            else:
                # Fallback for non-Postgres checkpointers (InMemorySaver)
                return {
                    "success": True,
                    "thread_id": thread_id,
                    "message": "Session marked for cleanup (in-memory session will be cleared on restart)"
                }
                
        except Exception as delete_error:
            print(f"Error during deletion: {delete_error}")
            raise HTTPException(status_code=500, detail=f"Error deleting session from database: {str(delete_error)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"API error in DELETE /session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


@app.get("/test/personas", response_model=PersonasListResponse)
def list_available_personas():
    try:
        print("API /test/personas - listing available personas")
        
        # Convert personas to PersonaInfo objects
        persona_infos = [
            PersonaInfo(
                name=p.name,
                description=p.description,
                sample_phrases=p.sample_phrases
            )
            for p in personas
        ]
        
        return PersonasListResponse(
            success=True,
            personas=persona_infos,
            total=len(persona_infos),
            message="Available test personas retrieved successfully"
        )
        
    except Exception as e:
        print(f"API error in /test/personas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving personas: {str(e)}")


@app.post("/test/persona")
def test_with_persona(request: TestPersonaRequest):
    try:
        print(f"API /test/persona - persona: {request.persona_name}, concept: {request.concept_title}")
        
        # Validate persona name (optional - warn if not in predefined list)
        available_persona_names = [p.name for p in personas]
        if request.persona_name not in available_persona_names:
            print(f"⚠️  Warning: '{request.persona_name}' is not a predefined persona. Available: {available_persona_names}")
        
        # Create session with persona
        start_request = StartSessionRequest(
            concept_title=request.concept_title,
            persona_name=request.persona_name,
            session_label=f"test-{request.persona_name.lower().replace(' ', '-')}"
        )
        
        return start_session(start_request)
        
    except Exception as e:
        print(f"API error in /test/persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating test session: {str(e)}")


@app.post("/test/images", response_model=TestImageResponse)
def get_test_image(request: TestImageRequest):
    try:
        print(f"API /test/images - concept: {request.concept_title}")
        
        selected_image = select_most_relevant_image_for_concept_introduction( #Same function is called by the agent when selecting images
            concept=request.concept_title,
            definition_context=request.definition_context or f"Learning about {request.concept_title}"
        )
        
        if selected_image:
            return TestImageResponse(
                success=True,
                concept=request.concept_title,
                image_url=selected_image.get("url"),
                image_description=selected_image.get("description", ""),
                message=f"Image retrieved successfully for '{request.concept_title}'"
            )
        else:
            return TestImageResponse(
                success=False,
                concept=request.concept_title,
                image_url=None,
                image_description=None,
                message=f"No image found for concept '{request.concept_title}'"
            )
        
    except Exception as e:
        print(f"API error in /test/images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving image: {str(e)}")


@app.post("/test/simulation", response_model=TestSimulationResponse)
def get_test_simulation(request: TestSimulationRequest):
    try:
        print(f"API /test/simulation - concept: {request.concept_title}, type: {request.simulation_type}")
        
        # Example variables for pendulum
        example_variables = [
            {"name": f"{request.simulation_type}", "role": "Independent Variable", "note": f"{request.simulation_type} of pendulum string"},
        ]
        
        # Create simulation config using the simulation_type from user input
        simulation_config = create_simulation_config( #Same function is called by the agent when simulation config needed
            variables=example_variables,
            concept=request.concept_title,
        )
        
        return TestSimulationResponse(
            success=True,
            concept=request.concept_title,
            simulation_config=simulation_config,
            message=f"Simulation config retrieved successfully for '{request.concept_title}'"
        )
        
    except Exception as e:
        print(f"API error in /test/simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving simulation config: {str(e)}")


@app.post("/concept-map/generate", response_model=ConceptMapResponse)
def generate_concept_map(request: ConceptMapRequest):
    """
    Generate concept map timeline from educational description.
    
    This endpoint:
    1. Accepts educational text description
    2. Extracts key concepts using Google Gemini AI
    3. Calculates character-based reveal times for each concept (0.08s per character)
    4. Saves complete timeline JSON to concept_json_timings/ folder
    5. Returns both the filepath and complete timeline data
    
    The generated JSON includes:
    - Concepts with reveal_time values (when to show each concept)
    - Relationships between concepts
    - Word-level timings (character-based)
    - Metadata (duration, concept count, etc.)
    """
    try:
        print(f"API /concept-map/generate - {len(request.description)} chars, level: {request.educational_level}")
        
        # Validate description is not empty
        if len(request.description.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Description cannot be empty"
            )
        
        # Use wrapper to set GOOGLE_API_KEY from GOOGLE_API_KEY_1
        with use_google_api_key():
            # Track API usage for concept map generation
            api_key_1 = os.getenv('GOOGLE_API_KEY_1')
            if api_key_1:
                # Concept map uses gemini-2.0-flash-lite model (hardcoded in timeline_mapper.py)
                track_model_call(api_key_1, "gemini-2.0-flash-lite")
                print(f"🔑 Tracked API usage for GOOGLE_API_KEY_1 (model: gemini-2.0-flash-exp)")
            
            # Step 1: Create timeline using concept_map_poc function
            # This calls Gemini API, calculates timings, and assigns reveal times
            print("🔄 Creating timeline with concept_map_poc...")
            timeline = create_timeline(
                description=request.description,
                educational_level=request.educational_level,
                topic_name=request.topic_name or ""
            )
        
        if not timeline:
            raise HTTPException(
                status_code=500,
                detail="Failed to create timeline. Check logs for details."
            )
        
        concepts_count = len(timeline.get('concepts', []))
        duration = timeline.get('metadata', {}).get('total_duration', 0.0)
        print(f"✅ Timeline created: {concepts_count} concepts, {duration:.1f}s duration")
        
        filepath = "Not Saved"

        # Step 2: Save JSON to disk (concept_json_timings/ folder)
        # print("💾 Saving timeline to disk...")
        # filepath = save_timeline_json_to_disk(timeline)
        
        # if not filepath:
        #     raise HTTPException(
        #         status_code=500,
        #         detail="Failed to save timeline JSON to disk"
        #     )
        
        # print(f"✅ Saved to: {filepath}")
        
        # Return success response with filepath and timeline data
        return ConceptMapResponse(
            success=True,
            filepath=filepath,
            timeline=timeline
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"API error in /concept-map/generate: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating concept map: {str(e)}")


print("=" * 80)
print("🎓 Educational Agent API Server Starting...")
print("=" * 80)
print(f"Agent Type: educational_agent_optimized_langsmith")
print(f"Concept: Dynamic (passed via API request)")
print(f"Persistence: Supabase-Postgres (LangGraph)")
print("=" * 80)
print("Available Endpoints:")
print("  GET  / - API information")
print("  GET  /health - Health check")
print("  GET  /concepts - List all available concepts")
print("  POST /session/start - Start new learning session")
print("  POST /session/continue - Continue existing session")
print("  GET  /session/status/{thread_id} - Get session status")
print("  GET  /session/history/{thread_id} - Get conversation history")
print("  GET  /session/summary/{thread_id} - Get session summary")
print("  DELETE /session/{thread_id} - Delete session")
print("  GET  /test/personas - List available test personas")
print("  POST /test/persona - Test with predefined persona")
print("  POST /test/images - Get image for a concept")
print("  POST /test/simulation - Get simulation config for a concept")
print("  POST /concept-map/generate - Generate concept map timeline (character-based timing)")
print("=" * 80)
print(f"Available Test Personas: {len(personas)}")
for p in personas:
    print(f"  - {p.name}: {p.description}")
print("=" * 80)
print("Starting server on http://0.0.0.0:8000")
print("API Docs available at http://localhost:8000/docs")
print("=" * 80)

# uvicorn.run(app, host="0.0.0.0", port=8000)


###To Do:
#1. ✅ Write postgres specific deletion logic - DONE
#2. ✅ Remove /sessions endpoint - DONE (useless endpoint removed)
#3. How will the test personas talk to agent via endpoint? Right now we have to call continue everytime after starting a session with a persona.