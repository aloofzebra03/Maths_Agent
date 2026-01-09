"""
Test script for the Math Tutoring Agent.

Demonstrates how to:
1. Initialize the agent with a problem
2. Interact with the agent using human-in-the-loop pattern
3. Track the conversation flow

The graph uses interrupt_after=["START", "ADAPTIVE_SOLVER"] which means:
- After START: graph pauses, waits for student response
- After ADAPTIVE_SOLVER: graph pauses, waits for student response
- Student responses are added via invoke() with HumanMessage
"""

from langchain_core.messages import HumanMessage
from educational_agent_math_tutor import graph, MathAgentState


def print_last_ai_message(result):
    """Helper to print the last AI message from the result."""
    for msg in reversed(result["messages"]):
        if hasattr(msg, 'content') and msg.__class__.__name__ == 'AIMessage':
            print(f"\n{msg.content}")
            return
    print("\n(No AI message found)")


def run_test_session():
    """
    Run a test tutoring session with the math agent using human-in-the-loop.
    """
    
    # Configuration
    thread_id = "test_session_1"
    problem_id = "add_frac_same_den_01"  # From Fractions.json
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    print("="*60)
    print("🎓 MATH TUTORING AGENT - TEST SESSION")
    print("="*60)
    print("📍 Using human-in-the-loop: graph will pause after START and ADAPTIVE_SOLVER")
    
    # ========================================================================
    # Step 1: START - Initialize with problem and present it to student
    # ========================================================================
    print(f"\n📚 Initializing session with problem: {problem_id}")
    
    initial_state = {
        "problem_id": problem_id,
        "messages": [],
    }
    
    # Invoke START node - will pause after START due to interrupt_after
    print("\n▶️ Invoking START node...")
    result = graph.invoke(initial_state, config)
    
    # Print agent's greeting
    print("\n" + "="*60)
    print("🤖 AGENT (after START):")
    print("="*60)
    print_last_ai_message(result)
    
    # Check if graph is interrupted (waiting for student input)
    if "__interrupt__" in result:
        print("\n⏸️ Graph interrupted - waiting for student response")
    
    # ========================================================================
    # Step 2: Student responds with their understanding
    # ========================================================================
    # ========================================================================
    # Step 2: Student responds with their understanding
    # ========================================================================
    print("\n" + "="*60)
    print("👨‍🎓 STUDENT RESPONSE #1:")
    print("="*60)
    student_response_1 = "I understand we need to add two fractions. I would add the numerators and denominators together."
    print(student_response_1)
    
    # Resume graph with student's message
    # This will run: ASSESSMENT → ADAPTIVE_SOLVER → interrupt
    print("\n▶️ Resuming graph with student response...")
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=student_response_1)]
        },
        config
    )
    
    # Print agent's assessment and pedagogical response
    print("\n" + "="*60)
    print("🤖 AGENT (after ASSESSMENT + ADAPTIVE_SOLVER):")
    print("="*60)
    print(f"📊 Mode Selected: {result.get('mode', 'unknown')}")
    print(f"📊 Ta (Approach): {result.get('Ta', 0.0):.2f}")
    print(f"📊 Tu (Understanding): {result.get('Tu', 0.0):.2f}")
    print_last_ai_message(result)
    
    # Check if graph is interrupted again
    if "__interrupt__" in result:
        print("\n⏸️ Graph interrupted - waiting for next student response")
    
    # ========================================================================
    # Step 3: Student tries again based on feedback
    # ========================================================================
    print("\n" + "="*60)
    print("👨‍🎓 STUDENT RESPONSE #2:")
    print("="*60)
    student_response_2 = "Oh, I see! So I should add the top numbers (3 + 2 = 5) and keep the bottom number the same (8). So the answer is 5/8."
    print(student_response_2)
    
    # Resume graph with student's corrected answer
    print("\n▶️ Resuming graph with student response...")
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=student_response_2)]
        },
        config
    )
    
    # Print agent's response
    print("\n" + "="*60)
    print("🤖 AGENT (after re-ASSESSMENT + ADAPTIVE_SOLVER):")
    print("="*60)
    print(f"✅ Problem Solved: {result.get('solved', False)}")
    print_last_ai_message(result)
    
    # If problem is solved, continue to REFLECTION
    if result.get('solved', False):
        print("\n▶️ Problem solved! Continuing to REFLECTION...")
        result = graph.invoke(None, config)  # Continue with no new input
        
        print("\n" + "="*60)
        print("🤖 AGENT (REFLECTION):")
        print("="*60)
        print_last_ai_message(result)
    
    # Print session summary
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(f"Total messages: {len(result['messages'])}")
    print(f"Final mode: {result.get('mode', 'unknown')}")
    print(f"Problem solved: {result.get('solved', False)}")
    print(f"Node transitions: {len(result.get('node_transitions', []))}")
    
    print("\n✅ Test session complete!")


if __name__ == "__main__":
    run_test_session()
