"""
Simple Streamlit UI for Math Tutoring Agent

Features:
- Problem selection from problems_json folder
- Text and image input support
- Chat-based interaction
- Session state debugging
- Reset functionality
"""

import streamlit as st
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from educational_agent_math_tutor.graph import graph
from langchain_core.messages import HumanMessage


# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="Math Tutor Agent",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for larger chat font
st.markdown("""
<style>
    /* Increase font size for chat messages */
    .stChatMessage {
        font-size: 1.1rem !important;
    }
    
    /* Increase font size for chat message content */
    .stChatMessage p {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Increase font size for all text in chat message container */
    [data-testid="stChatMessageContent"] {
        font-size: 1.1rem !important;
    }
    
    [data-testid="stChatMessageContent"] p {
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Problem Loading Utilities
# ============================================================================

def load_all_problems() -> Dict[str, Dict[str, str]]:
    """
    Load all problems from problems_json directory.
    
    Returns:
        Dict mapping problem_id to {topic, question, difficulty}
    """
    problems = {}
    
    # Get path to problems_json directory
    current_dir = Path(__file__).parent.parent
    problems_path = current_dir / "problems_json"
    
    if not problems_path.exists():
        st.error(f"Problems directory not found: {problems_path}")
        return problems
    
    # Scan all JSON files
    for filepath in problems_path.glob("*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse newline-delimited JSON objects
            json_objects = []
            current_obj = ""
            brace_count = 0
            
            for char in content:
                current_obj += char
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and current_obj.strip():
                        try:
                            obj = json.loads(current_obj.strip())
                            json_objects.append(obj)
                            current_obj = ""
                        except json.JSONDecodeError:
                            pass
            
            # Extract problem metadata
            for problem_data in json_objects:
                if 'problem_id' in problem_data:
                    problems[problem_data['problem_id']] = {
                        'topic': problem_data.get('topic', 'Unknown Topic'),
                        'question': problem_data.get('question', 'No question text'),
                        'difficulty': problem_data.get('difficulty', 'unknown')
                    }
                    
        except Exception as e:
            st.warning(f"Error reading {filepath.name}: {e}")
            continue
    
    return problems


def convert_image_to_base64(uploaded_file) -> str:
    """
    Convert uploaded image to base64 data URI.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Base64 data URI string
    """
    # Get file extension
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    # Read file bytes
    img_bytes = uploaded_file.getvalue()
    
    # Encode to base64
    img_base64 = base64.b64encode(img_bytes).decode()
    
    # Create data URI
    data_uri = f"data:image/{file_ext};base64,{img_base64}"
    
    return data_uri


# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if "session_started" not in st.session_state:
        st.session_state.session_started = False
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if "selected_problem_id" not in st.session_state:
        st.session_state.selected_problem_id = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    
    if "uploaded_image_data" not in st.session_state:
        st.session_state.uploaded_image_data = None


def reset_session():
    """Reset session state to start a new problem."""
    st.session_state.session_started = False
    st.session_state.session_id = None
    st.session_state.selected_problem_id = None
    st.session_state.messages = []
    st.session_state.processing = False
    st.session_state.last_result = None
    st.session_state.uploaded_image_data = None


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application."""
    
    initialize_session_state()
    
    # Load all available problems
    all_problems = load_all_problems()
    
    if not all_problems:
        st.error("No problems found in problems_json directory!")
        st.stop()
    
    # ========================================================================
    # Start Screen (Problem Selection)
    # ========================================================================
    
    if not st.session_state.session_started:
        st.title("🧮 Math Tutor Agent")
        st.markdown("---")
        
        st.subheader("Select a Problem")
        
        # Problem selection dropdown
        problem_ids = sorted(all_problems.keys())
        selected_id = st.selectbox(
            "Choose a problem to work on:",
            options=problem_ids,
            format_func=lambda x: x
        )
        
        # Display selected problem details
        if selected_id:
            problem_info = all_problems[selected_id]
            st.markdown(f"""
            **Topic:** {problem_info['topic']}  
            **Difficulty:** {problem_info['difficulty'].capitalize()}  
            **Question:** {problem_info['question']}
            """)
        
        # Start button
        if st.button("Start Session", type="primary"):
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.selected_problem_id = selected_id
            st.session_state.session_started = True
            st.session_state.processing = True
            st.rerun()
        
        st.stop()
    
    # ========================================================================
    # Active Session Interface
    # ========================================================================
    
    # Sidebar with debugging info and controls
    with st.sidebar:
        st.title("🧮 Math Tutor")
        
        # Current problem display
        if st.session_state.selected_problem_id:
            problem_info = all_problems.get(st.session_state.selected_problem_id, {})
            st.markdown("### Current Problem")
            st.info(f"**{problem_info.get('question', 'Unknown')}**")
            st.caption(f"Topic: {problem_info.get('topic', 'Unknown')}")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 New Problem", use_container_width=True):
            reset_session()
            st.rerun()
        
        st.markdown("---")
        
        # Debug information
        st.markdown("### Debug Info")
        
        if st.session_state.last_result:
            result = st.session_state.last_result
            
            st.metric("Current State", result.get("current_state", "Unknown"))
            st.metric("Mode", result.get("mode", "Unknown"))
            st.metric("Solved", "✅ Yes" if result.get("solved", False) else "❌ No")
            
            col1, col2 = st.columns(2)
            with col1:
                ta_score = result.get("Ta", 0.0)
                st.metric("Ta Score", f"{ta_score:.2f}")
            with col2:
                tu_score = result.get("Tu", 0.0)
                st.metric("Tu Score", f"{tu_score:.2f}")
            
            # Step progress (for scaffold mode)
            if result.get("steps"):
                step_idx = result.get("step_index", 0)
                max_steps = result.get("max_steps", len(result.get("steps", [])))
                st.metric("Step Progress", f"{step_idx + 1} / {max_steps}")
        else:
            st.caption("Waiting for first response...")
    
    # ========================================================================
    # Processing Phase (Two-Phase Rerun Pattern)
    # ========================================================================
    
    if st.session_state.processing:
        st.session_state.processing = False
        
        config = {
            "configurable": {
                "thread_id": st.session_state.session_id
            }
        }
        
        with st.spinner("Thinking..."):
            try:
                # Check if this is the initial call or continuation
                if not st.session_state.messages:
                    # Initial call - start the session
                    initial_state = {
                        "problem_id": st.session_state.selected_problem_id,
                        "messages": []
                    }
                    result = graph.invoke(initial_state, config)
                else:
                    # Continuation - get last user message
                    last_user_msg = None
                    for role, msg in reversed(st.session_state.messages):
                        if role == "user":
                            last_user_msg = msg
                            break
                    
                    if last_user_msg:
                        result = graph.invoke(
                            {"messages": [HumanMessage(content=last_user_msg)]},
                            config
                        )
                    else:
                        st.error("No user message found to process")
                        st.stop()
                
                # Store result for debugging
                st.session_state.last_result = result
                
                # Extract AI response
                ai_response = result.get("agent_output", "No response from agent")
                
                # Append assistant message
                st.session_state.messages.append(("assistant", ai_response))
                
            except Exception as e:
                st.error(f"Error communicating with agent: {e}")
                st.exception(e)
        
        st.rerun()
    
    # ========================================================================
    # Chat Display
    # ========================================================================
    
    st.title("💬 Chat")
    
    # Display all messages
    for i, (role, content) in enumerate(st.session_state.messages):
        with st.chat_message(role):
            # Check if content is a base64 image
            if isinstance(content, str) and content.startswith("data:image/"):
                # Display image thumbnail
                st.image(content, width=300)
                st.caption("📷 Uploaded image")
            else:
                st.write(content)
    
    # ========================================================================
    # User Input
    # ========================================================================
    
    # Check if session is complete
    session_complete = False
    if st.session_state.last_result:
        if st.session_state.last_result.get("current_state") == "REFLECTION":
            session_complete = True
    
    if session_complete:
        st.success("🎉 Session complete! Click 'New Problem' to start another session.")
    else:
        # Text input
        text_input = st.chat_input("Type your response here...")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Or upload an image of your work:",
            type=['png', 'jpg', 'jpeg', 'webp', 'gif'],
            key=f"image_upload_{len(st.session_state.messages)}"
        )
        
        # Process text input
        if text_input:
            # Append user message
            st.session_state.messages.append(("user", text_input))
            st.session_state.processing = True
            st.rerun()
        
        # Process image upload
        elif uploaded_file:
            # Convert to base64 data URI
            data_uri = convert_image_to_base64(uploaded_file)
            
            # Append user message (image)
            st.session_state.messages.append(("user", data_uri))
            st.session_state.processing = True
            st.rerun()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
