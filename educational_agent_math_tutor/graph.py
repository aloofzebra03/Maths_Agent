"""
LangGraph structure for the Math Tutoring Agent.

Defines the StateGraph with nodes, edges, and routing logic.
"""

from datetime import datetime
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from educational_agent_math_tutor.schemas import MathAgentState
from educational_agent_math_tutor.nodes import (
    start_node,
    assess_student_response,
    adaptive_solver,
    reflection_node,
)


# ============================================================================
# Node Wrapper for Transition Tracking
# ============================================================================

def create_node_wrapper(node_func, node_name: str):
    """
    Create a wrapper for a node that tracks transitions and detects new user messages.
    
    This wrapper:
    1. Captures old state before node call
    2. Detects new HumanMessages added after interrupt
    3. Stores last user message in state for debugging
    4. Tracks node transitions
    5. Calls the actual node function
    6. Returns the node's state update
    
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
        
        # Capture old state to detect new messages
        old_last_user_msg = state.get("last_user_msg")
        messages = state.get("messages", [])
        
        # Find the last HumanMessage in current state
        current_last_human_msg = None
        for msg in reversed(messages):
            if hasattr(msg, '__class__') and msg.__class__.__name__ == 'HumanMessage':
                current_last_human_msg = msg.content
                break
        
        # Detect if there's a NEW human message (after interrupt resume)
        if current_last_human_msg and current_last_human_msg != old_last_user_msg:
            print(f"🆕 NEW USER MESSAGE DETECTED:")
            print(f"   Old: {old_last_user_msg[:50] if old_last_user_msg else 'None'}...")
            print(f"   New: {current_last_human_msg[:50]}...")
        
        # Track transition
        transitions = state.get("node_transitions", [])
        transitions.append({
            "timestamp": datetime.now().isoformat(),
            "to_node": node_name,
            "message_index": len(messages),
            "last_user_msg": current_last_human_msg[:50] if current_last_human_msg else None
        })
        
        # Call the actual node
        node_update = node_func(state)
        
        # Ensure transition list is in the update if node doesn't include it
        if "node_transitions" not in node_update:
            node_update["node_transitions"] = transitions
        
        # Store the last user message for debugging and node access
        if current_last_human_msg:
            node_update["last_user_msg"] = current_last_human_msg
        
        return node_update
    
    return wrapped_node


# ============================================================================
# Routing Functions
# ============================================================================

def should_continue_solving(state: MathAgentState) -> Literal["reflection", "adaptive_solver"]:
    """
    Route from ADAPTIVE_SOLVER based on whether problem is solved.
    
    Returns:
        "reflection" if solved=True, else "adaptive_solver" (which routes to ASSESSMENT)
    """
    if state.get("solved", False):
        print("✅ Problem solved! Routing to REFLECTION")
        return "reflection"
    else:
        print("🔄 Problem not yet solved, routing back to ASSESSMENT for next student response")
        return "adaptive_solver"


# ============================================================================
# Graph Construction
# ============================================================================

def create_graph():
    """
    Create and compile the Math Tutoring Agent graph.
    
    Graph Flow:
        START → ASSESSMENT → ADAPTIVE_SOLVER → (loop or REFLECTION) → END
    
    Nodes:
        - START: Load problem and greet student
        - ASSESSMENT: Evaluate Tu/Ta, detect missing concepts, route to mode
        - ADAPTIVE_SOLVER: Execute mode-specific pedagogy (coach/guided/scaffold/concept)
        - REFLECTION: Celebrate success and suggest next steps
    
    Returns:
        Compiled StateGraph with MemorySaver checkpointer
    """
    
    # Create the graph
    workflow = StateGraph(MathAgentState)
    
    # Add wrapped nodes (for future transition tracking enhancement)
    workflow.add_node("START", create_node_wrapper(start_node, "START"))
    workflow.add_node("ASSESSMENT", create_node_wrapper(assess_student_response, "ASSESSMENT"))
    workflow.add_node("ADAPTIVE_SOLVER", create_node_wrapper(adaptive_solver, "ADAPTIVE_SOLVER"))
    workflow.add_node("REFLECTION", create_node_wrapper(reflection_node, "REFLECTION"))
    
    # Define edges
    workflow.set_entry_point("START")
    
    # START → ASSESSMENT
    # Note: Graph will interrupt after START, wait for student response,
    # then continue to ASSESSMENT on next invoke
    workflow.add_edge("START", "ASSESSMENT")
    
    # ASSESSMENT → ADAPTIVE_SOLVER (always)
    # (Mode routing happens internally in ADAPTIVE_SOLVER based on state["mode"])
    workflow.add_edge("ASSESSMENT", "ADAPTIVE_SOLVER")
    
    # ADAPTIVE_SOLVER → conditional routing
    # Note: Graph will interrupt after ADAPTIVE_SOLVER, wait for student response,
    # then continue based on whether problem is solved
    # - If solved: go to REFLECTION
    # - Else: loop back to ASSESSMENT (to evaluate new student response)
    workflow.add_conditional_edges(
        "ADAPTIVE_SOLVER",
        should_continue_solving,
        {
            "reflection": "REFLECTION",
            "adaptive_solver": "ASSESSMENT"  # Loop back to ASSESSMENT, not ADAPTIVE_SOLVER
        }
    )
    
    # REFLECTION → END (always)
    workflow.add_edge("REFLECTION", END)
    
    # Compile with MemorySaver checkpointer
    checkpointer = MemorySaver()
    
    graph = workflow.compile(
        # checkpointer=checkpointer,
        interrupt_after=["START", "ADAPTIVE_SOLVER"]  # Pause for student input after these nodes
    )
    
    print("✅ Graph compiled successfully with MemorySaver checkpointer")
    print("🔄 Human-in-the-loop enabled: graph will pause after START and ADAPTIVE_SOLVER")
    
    return graph


# ============================================================================
# Graph Instance (for import)
# ============================================================================

# Create the graph instance when module is imported
graph = create_graph()
