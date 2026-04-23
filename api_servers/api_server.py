from fastapi import FastAPI, HTTPException, status, Form, File, UploadFile
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, Optional, Any, List
import sys
import os
import traceback
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path to import educational agent
# sys.path.insert(0, str(Path(__file__).parent.parent))

from educational_agent_math_tutor.graph import graph
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from api_servers_reference.schemas import (
    StartSessionRequest, StartSessionResponse,
    ContinueSessionResponse,
    ProblemInfo,
    ProblemsListResponse,
)
from api_tracker_utils.error import APITrackerError, MinuteLimitExhaustedError, DayLimitExhaustedError
from utils.ocr_utilities import process_image_from_path


MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _iter_json_objects(raw_text: str):
    """Yield JSON objects from text that may contain concatenated JSON blobs."""
    decoder = json.JSONDecoder()
    idx = 0
    length = len(raw_text)

    while idx < length:
        while idx < length and raw_text[idx].isspace():
            idx += 1
        if idx >= length:
            break

        try:
            obj, next_idx = decoder.raw_decode(raw_text, idx)
            yield obj
            idx = next_idx
        except json.JSONDecodeError:
            # Skip one character and keep scanning to avoid hard failure on malformed segments.
            idx += 1


def _load_problem_catalog() -> List[Dict[str, Any]]:
    """Load problem metadata from problems_json files."""
    problems_dir = Path(__file__).parent.parent / "problems_json"
    if not problems_dir.exists():
        return []

    catalog: List[Dict[str, Any]] = []
    for json_file in sorted(problems_dir.glob("*.json")):
        try:
            content = json_file.read_text(encoding="utf-8")
            for obj in _iter_json_objects(content):
                if not isinstance(obj, dict):
                    continue
                pid = obj.get("problem_id")
                topic = obj.get("topic")
                if isinstance(pid, str) and pid.strip() and isinstance(topic, str) and topic.strip():
                    catalog.append(
                        {
                            "problem_id": pid.strip(),
                            "topic": topic.strip(),
                            "difficulty": obj.get("difficulty"),
                        }
                    )
        except Exception as read_error:
            print(f"[WARN] Failed loading problem catalog from {json_file.name}: {read_error}")

    return catalog


def _resolve_problem(problem_id: Optional[str]) -> Dict[str, Any]:
    """Resolve incoming session target to a concrete problem from catalog using problem_id."""
    catalog = _load_problem_catalog()
    if not catalog:
        raise HTTPException(status_code=500, detail="No problems available in problems_json")

    if problem_id:
        for item in catalog:
            if item["problem_id"].lower() == problem_id.strip().lower():
                return item
        raise HTTPException(status_code=400, detail=f"Unknown problem_id '{problem_id}'")

    raise HTTPException(status_code=400, detail="Provide problem_id")


def _validate_uploaded_image(image: UploadFile) -> None:
    """Validate uploaded image extension and size before processing."""
    file_extension = os.path.splitext(image.filename or "")[1].lower()
    if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{file_extension}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}"
        )

    # Preserve stream position while checking file size.
    current_pos = image.file.tell()
    image.file.seek(0, os.SEEK_END)
    file_size = image.file.tell()
    image.file.seek(current_pos)

    if file_size > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({file_size} bytes). Max allowed: {MAX_IMAGE_SIZE_BYTES} bytes"
        )


def _save_uploaded_image_to_temp(image: UploadFile) -> Path:
    """Persist uploaded image to temp directory and return the saved path."""
    upload_dir = Path("./temp/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_extension = os.path.splitext(image.filename or "")[1] or ".jpg"
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_path = upload_dir / temp_filename

    file_bytes = image.file.read()
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    return temp_path

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

def generate_thread_id(problem_title: str, is_kannada: bool = False, label: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """
    Generate a unique thread ID with ordered components for better organization.
    
    Args:
        problem_title: The problem/topic being taught
        is_kannada: Whether the session is in Kannada
        label: Optional custom session label
        user_id: Optional user/student ID
    
    Returns:
        Formatted thread ID: <user_id>-<label>-<problem>-<lang>-thread-<timestamp>
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Build parts in order: user_id, label, problem, language
    parts = []
    
    # 1. User ID (if present)
    if user_id:
        parts.append(user_id)
    
    # 2. Label (if present)
    if label:
        parts.append(label)
    
    # 3. Problem (always present, cleaned)
    problem_slug = problem_title.lower().replace(" ", "-").replace("'", "").replace(",", "")
    parts.append(problem_slug)
    
    # 4. Language indicator
    lang = "kannada" if is_kannada else "english"
    parts.append(lang)
    
    # If no user_id or label, add "session" at the beginning
    if not user_id and not label:
        parts.insert(0, "session")
    
    # Join all parts with thread indicator and timestamp
    return f"{'-'.join(parts)}-thread-{timestamp}"


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
        image_obj = enhanced_meta.get("image")
        if isinstance(image_obj, dict):
            image_description = image_obj.get("description")
            image_url = image_obj.get("url")
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
        "message": "Educational Maths Agent API is running!",
        "version": "1.0.0",
        "agent_type": "educational_agent_math_tutor",
        "endpoints": [
            "GET  /problems - List all available problems",
            "POST /session/start - Start new learning session",
            "POST /session/continue - Continue existing session (multipart: text and/or image)",
        ]
    }


@app.get("/problems", response_model=ProblemsListResponse)
def list_available_problems():
    """List all available problems that can be taught by the educational agent."""
    try:
        print("API /problems - Retrieving all available problems")

        catalog = _load_problem_catalog()
        problems = [
            ProblemInfo(
                problem_id=item["problem_id"],
                topic=item["topic"],
                difficulty=item.get("difficulty") if isinstance(item.get("difficulty"), str) else None,
            )
            for item in catalog
        ]

        return ProblemsListResponse(
            success=True,
            problems=problems,
            total=len(problems),
            message=f"Retrieved {len(problems)} available problems",
        )

    except Exception as e:
        print(f"API error in /problems: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving problems: {str(e)}")


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(request: StartSessionRequest):
    try:
        resolved_problem = _resolve_problem(request.problem_id)
        selected_problem_id = resolved_problem["problem_id"]
        selected_topic = resolved_problem["topic"]

        print(
            f"API /session/start - problem_id: {selected_problem_id}, topic: {selected_topic}, "
            f"student: {request.student_id}, language: {'Kannada' if request.is_kannada else 'English'}"
        )
        
        # Generate unique thread_id with problem and language info
        thread_id = generate_thread_id(
            problem_title=selected_topic,
            is_kannada=request.is_kannada,
            label=request.session_label,
            user_id=request.student_id
        )
        
        # Generate session_id and user_id
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = request.session_label or "session"
        session_id = f"{base}-{timestamp}"
        user_id = request.student_id or "anonymous"
        
        print(f"[INFO] Generated thread_id: {thread_id}")
        
        # # Validate model if provided
        # model = request.model or "gemma-3-27b-it"
        # if model not in AVAILABLE_GEMINI_MODELS:
        #     raise HTTPException(
        #         status_code=400,
        #         detail=f"Invalid model '{model}'. Available models: {AVAILABLE_GEMINI_MODELS}"
        #     )
        
        # Start the conversation by invoking the graph with __start__ message
        # Tracker will automatically select the best API key and model based on rate limits
        print("Invoking graph to start session (tracker will select optimal model)")
        result = graph.invoke(
            {
                "messages": [HumanMessage(content="__start__")],
                "is_kannada": request.is_kannada,
                "concept_title": selected_topic,
                "problem_id": selected_problem_id,
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
        
        return StartSessionResponse(
            success=True,
            session_id=session_id,
            thread_id=thread_id,
            problem_id=selected_problem_id,
            user_id=user_id,
            agent_response=agent_response,
            current_state=result.get("current_state", "START"),
            message="Session started successfully. Agent is ready for student input.",
            metadata=metadata
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

    except APITrackerError as e:
        print(f"[API] Tracker error: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=503,
            detail=f"Tracker error: {e}"
        )
    
    except Exception as e:
        print(f"API error in /session/start: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")


@app.post("/session/continue", response_model=ContinueSessionResponse)
def continue_session(
    thread_id: str = Form(...),
    user_message: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    temp_image_path: Optional[Path] = None
    try:
        resolved_thread_id = thread_id.strip()
        resolved_user_message = user_message.strip()
        has_image = bool(image and image.filename)

        if not has_image and not resolved_user_message:
            raise HTTPException(status_code=400, detail="Provide either image or user_message")

        input_content = resolved_user_message
        if has_image and image:
            _validate_uploaded_image(image)
            temp_image_path = _save_uploaded_image_to_temp(image)
            ocr_result = process_image_from_path(str(temp_image_path))
            image_text = (ocr_result.get("text") or "").strip()
            if not image_text:
                image_text = "[Image uploaded but no text extracted.]"

            if resolved_user_message:
                input_content = f"{image_text}\n\n{resolved_user_message}"
            else:
                input_content = image_text

            print(f"[INFO] Image uploaded: {image.filename} -> {temp_image_path}")

        print(f"API /session/continue - thread: {resolved_thread_id}, message: {input_content[:50]}...")
        
        # Check if session exists by trying to get its state
        existing_state = get_state_from_checkpoint(resolved_thread_id)
        if existing_state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found for thread_id: {resolved_thread_id}. Please start a new session."
            )
        
        update_dict = {
            "messages": [HumanMessage(content=input_content)]
        }
        
        # Continue the conversation using Command (resume)
        cmd = Command(
            resume=True,
            update=update_dict,
        )
        
        # Invoke graph with the user message
        result = graph.invoke(
            cmd,
            config={"configurable": {"thread_id": resolved_thread_id}},
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
        
        return ContinueSessionResponse(
            success=True,
            thread_id=resolved_thread_id,
            agent_response=agent_response,
            current_state=result.get("current_state", "UNKNOWN"),
            metadata=metadata,
            message="Response generated successfully"
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

    except APITrackerError as e:
        print(f"[API] Tracker error: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=503,
            detail=f"Tracker error: {e}"
        )
        
    except HTTPException:
        raise

    except Exception as e:
        print(f"API error in /session/continue: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error continuing session: {str(e)}")
    finally:
        if temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
            except Exception as cleanup_error:
                print(f"[WARN] Failed to cleanup temp upload {temp_image_path}: {cleanup_error}")


print("=" * 80)
print("Educational Agent API Server Starting...")
print("=" * 80)
print(f"Agent Type: educational_agent_optimized_langsmith")
print(f"Problem: Dynamic (passed via API request)")
print(f"Persistence: Supabase-Postgres (LangGraph)")
print("=" * 80)
print("Available Endpoints:")
print("  GET  / - API information")
print("  GET  /problems - List all available problems")
print("  POST /session/start - Start new learning session")
print("  POST /session/continue - Continue existing session (multipart: text and/or image)")
print("=" * 80)
print("Starting server on http://0.0.0.0:8000")
print("API Docs available at http://localhost:8000/docs")
print("=" * 80)

# uvicorn.run(app, host="0.0.0.0", port=8000)