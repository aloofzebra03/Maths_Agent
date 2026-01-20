"""
LangGraph structure for the Math Tutoring Agent.

Defines the StateGraph with nodes, edges, and routing logic.
"""

from datetime import datetime
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from educational_agent_math_tutor.schemas import MathAgentState
from educational_agent_math_tutor.nodes import (
    start_node,
    assess_student_response,
    concept_node,
    re_ask_start_questions_node,
    assess_approach_node,
    adaptive_solver,
    reflection_node,
)
from educational_agent_math_tutor.input_processor import detect_and_process_input


# ============================================================================
# Node Wrapper for Transition Tracking
# ============================================================================

def create_node_wrapper(node_func, node_name: str):
    """
    Create a wrapper for a node that tracks transitions and detects new user messages.
    
    This wrapper (following reference implementation pattern):
    1. Captures old state before node call
    2. Detects new HumanMessages and MUTATES state directly
    3. Calls the node function
    4. Merges node updates into state
    5. Tracks transitions
    6. Returns the mutated state object
    
    Future Enhancement:
    When adding memory optimization later, you can enhance this wrapper to:
    - Build conversation summaries
    - Populate summary and summary_last_index fields
    
    Args:
        node_func: The actual node function to wrap
        node_name: Name of the node (for tracking)
    
    Returns:
        Wrapped node function
    """
    def wrapped_node(state: MathAgentState):
        print(f"\n{'='*60}")
        print(f"🔄 TRANSITION TO: {node_name}")
        print(f"{'='*60}")
        print(f"📊 Messages count: {len(state.get('messages', []))}")
        
        # CAPTURE OLD STATE
        old_state = state.get("current_state")
        old_last_user_msg = state.get("last_user_msg")
        
        # DETECT AND STORE NEW USER MESSAGE (mutate state directly)
        print(old_last_user_msg)
        messages = state.get("messages", [])
        # print(messages)
        if messages and isinstance(messages[-1], HumanMessage):
            raw_content = messages[-1].content or ""
            
            # PROCESS MULTIMODAL INPUT (image/audio → text)
            processed = detect_and_process_input(raw_content)
            # print(f"############### Processed input: {processed['processed_text']}")
            processed_text = processed["processed_text"]
            
            # If input was an image, update the message content with extracted text
            if processed["input_type"] in ["image_path", "image_base64"]:
                print(f"🖼️ Image input detected: {processed['input_type']}")
                print(f"📝 OCR extracted: {processed_text[:100]}...")
                # Replace image data/path with extracted text in message
                messages[-1].content = processed_text
                text = processed_text
            else:
                text = raw_content
            
            if text and text != old_last_user_msg:
                print(f"🆕 NEW USER MESSAGE DETECTED:")
                print(f"   Old: {old_last_user_msg[:50] if old_last_user_msg else 'None'}...")
                print(f"   New: {text[:50]}...")
                state["last_user_msg"] = text  # ✅ MUTATE state directly
            else:
                print("ℹ️ No change in last_user_msg")
        else:
            print("ℹ️ Last message is not HumanMessage")
        
        # CALL THE NODE FUNCTION
        node_update = node_func(state)
        
        # MERGE NODE UPDATE INTO STATE (reference pattern)
        state.update(node_update)
        
        # CAPTURE NEW STATE AFTER PROCESSING
        new_state = state.get("current_state")
        final_message_count = len(state.get("messages", []))
        
        # TRACK TRANSITION IF STATE CHANGED
        if old_state != new_state and old_state is not None:
            transitions = state.setdefault("node_transitions", [])
            transitions.append({
                "from_node": old_state,
                "to_node": new_state,
                "transition_after_message_index": final_message_count,
                "timestamp": datetime.now().isoformat()
            })
            print(f"🔄 NODE TRANSITION: {old_state} -> {new_state} after message {final_message_count}")
        
        # RETURN THE MUTATED STATE OBJECT (reference pattern)
        return state
    
    return wrapped_node


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_initial_assessment(state: MathAgentState) -> Literal["concept", "assess_approach"]:
    """
    Route from ASSESSMENT based on whether student knows required concepts.
    
    Returns:
        "concept" if missing_concepts detected, else "assess_approach"
    """
    missing_concepts = state.get("missing_concepts", [])
    
    if missing_concepts:
        print(f"📚 Concepts missing: {missing_concepts}. Routing to CONCEPT node")
        return "concept"
    else:
        print("✅ All concepts understood. Routing to ASSESS_APPROACH")
        return "assess_approach"


def route_after_concept(state: MathAgentState) -> Literal["concept", "re_ask"]:
    """
    Route from CONCEPT node.
    
    Returns:
        "concept" - if student needs retry OR more concepts to teach (self-loop)
        "re_ask" - if all concepts taught
    """
    missing_concepts = state.get("missing_concepts", [])
    
    if missing_concepts:
        # Still have concepts to teach (or retrying current concept)
        print(f"📚 Remaining concepts: {missing_concepts}. Looping back to CONCEPT")
        return "concept"
    else:
        # All concepts taught
        print("✅ All concepts taught. Routing to RE_ASK")
        return "re_ask"


def should_continue_solving(state: MathAgentState) -> Literal["reflection", "assess_approach"]:
    """
    Route from ADAPTIVE_SOLVER based on whether problem is solved.
    
    Returns:
        "reflection" if solved=True, else "assess_approach" (to check progress)
    """
    if state.get("solved", False):
        print("✅ Problem solved! Routing to REFLECTION")
        return "reflection"
    else:
        print("🔄 Problem not yet solved, routing to ASSESS_APPROACH to check progress")
        return "assess_approach"


# ============================================================================
# Graph Construction
# ============================================================================

def create_graph():
    """
    Create and compile the Math Tutoring Agent graph.
    
    Graph Flow (Option 2 - Simplified):
        START → ASSESSMENT → [concept check]
                  ↓
        [Missing concepts?]
                  ↓ YES                    ↓ NO
              CONCEPT                  ASSESS_APPROACH
                  ↓                          ↓
              RE_ASK ──────────────────────→│
                  ↓
          ASSESS_APPROACH → ADAPTIVE_SOLVER → [loop or REFLECTION] → END
    
    Nodes:
        - START: Load problem and greet student with initial questions
        - ASSESSMENT: Check if student knows required concepts
        - CONCEPT: Teach missing concepts (standalone node, not a mode)
        - RE_ASK: Re-ask START questions after teaching concepts
        - ASSESS_APPROACH: Score Tu/Ta and route to pedagogical mode
        - ADAPTIVE_SOLVER: Execute mode-specific pedagogy (coach/guided/scaffold)
        - REFLECTION: Celebrate success and suggest next steps
    
    Returns:
        Compiled StateGraph with MemorySaver checkpointer
    """
    
    # Create the graph
    workflow = StateGraph(MathAgentState)
    
    # Add wrapped nodes
    workflow.add_node("START", create_node_wrapper(start_node, "START"))
    workflow.add_node("ASSESSMENT", create_node_wrapper(assess_student_response, "ASSESSMENT"))
    workflow.add_node("CONCEPT", create_node_wrapper(concept_node, "CONCEPT"))
    workflow.add_node("RE_ASK", create_node_wrapper(re_ask_start_questions_node, "RE_ASK"))
    workflow.add_node("ASSESS_APPROACH", create_node_wrapper(assess_approach_node, "ASSESS_APPROACH"))
    workflow.add_node("ADAPTIVE_SOLVER", create_node_wrapper(adaptive_solver, "ADAPTIVE_SOLVER"))
    workflow.add_node("REFLECTION", create_node_wrapper(reflection_node, "REFLECTION"))
    
    # Define edges
    workflow.set_entry_point("START")
    
    # START → ASSESSMENT
    # Graph will interrupt after START, wait for student response, then continue to ASSESSMENT
    workflow.add_edge("START", "ASSESSMENT")
    
    # ASSESSMENT → conditional (CONCEPT or ASSESS_APPROACH)
    # Based on whether student knows required concepts
    workflow.add_conditional_edges(
        "ASSESSMENT",
        route_after_initial_assessment,
        {
            "concept": "CONCEPT",
            "assess_approach": "ASSESS_APPROACH"
        }
    )
    
    # CONCEPT → conditional (CONCEPT or RE_ASK)
    # After teaching concepts, either loop back for more concepts/retries or proceed to RE_ASK
    workflow.add_conditional_edges(
        "CONCEPT",
        route_after_concept,
        {
            "concept": "CONCEPT",  # Self-loop for retries or next concept
            "re_ask": "RE_ASK"
        }
    )
    
    # RE_ASK → ASSESS_APPROACH (always)
    # After re-asking, assess the student's new response
    workflow.add_edge("RE_ASK", "ASSESS_APPROACH")
    
    # ASSESS_APPROACH → ADAPTIVE_SOLVER (always)
    # After scoring Tu/Ta, route to appropriate mode
    workflow.add_edge("ASSESS_APPROACH", "ADAPTIVE_SOLVER")
    
    # ADAPTIVE_SOLVER → conditional routing
    # Graph will interrupt after ADAPTIVE_SOLVER for student response
    # - If solved: go to REFLECTION
    # - Else: loop back to ASSESS_APPROACH
    workflow.add_conditional_edges(
        "ADAPTIVE_SOLVER",
        should_continue_solving,
        {
            "reflection": "REFLECTION",
            "assess_approach": "ASSESS_APPROACH"
        }
    )
    
    # REFLECTION → END (always)
    workflow.add_edge("REFLECTION", END)
    
    # Compile with MemorySaver checkpointer
    checkpointer = MemorySaver()
    
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["START", "ADAPTIVE_SOLVER", "RE_ASK", "CONCEPT"]  # Pause for student input
    )
    
    print("✅ Graph compiled successfully")
    print("🔄 Human-in-the-loop enabled at: START, CONCEPT, RE_ASK, ADAPTIVE_SOLVER")
    
    return graph


# ============================================================================
# Graph Instance (for import)
# ============================================================================

# Create the graph instance when module is imported
graph = create_graph()
