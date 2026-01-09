"""
Interactive demo of the Math Tutoring Agent with human-in-the-loop.

Run this script to have a real conversation with the agent.
The agent will pause after presenting the problem and after each teaching interaction,
waiting for you to type your response.
"""

from langchain_core.messages import HumanMessage
from educational_agent_math_tutor import graph


def print_separator():
    """Print a visual separator."""
    print("\n" + "="*70)


def print_ai_message(result):
    """Extract and print the last AI message."""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, 'content') and msg.__class__.__name__ == 'AIMessage':
            print(f"\n🤖 TUTOR:\n{msg.content}")
            return
    print("\n(No message from tutor)")


def main():
    """Run interactive tutoring session."""
    
    print_separator()
    print("🎓 INTERACTIVE MATH TUTORING SESSION")
    print_separator()
    
    # Setup
    thread_id = input("\n📝 Enter a session ID (e.g., 'session_1'): ").strip() or "session_1"
    problem_id = input("📚 Enter problem ID (default: 'add_frac_same_den_01'): ").strip() or "add_frac_same_den_01"
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize session
    print("\n🚀 Starting tutoring session...")
    initial_state = {
        "problem_id": problem_id,
        "messages": [],
    }
    
    # START node - present problem and pause
    result = graph.invoke(initial_state, config)
    print_separator()
    print_ai_message(result)
    print_separator()
    
    # Main conversation loop
    turn = 1
    while True:
        # Check if we've reached the end
        if result.get("solved", False) and result.get("current_state") == "REFLECTION":
            print("\n✅ Tutoring session complete!")
            break
        
        # Get student input
        print(f"\n👨‍🎓 YOUR TURN #{turn}:")
        user_input = input("Type your response (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Ending session. Goodbye!")
            break
        
        if not user_input:
            print("⚠️ Please enter a response.")
            continue
        
        # Send student response to agent
        print("\n⏳ Agent is thinking...")
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )
        
        # Display agent's response
        print_separator()
        
        # Show diagnostics
        mode = result.get("mode")
        if mode:
            print(f"📊 Mode: {mode.upper()}")
        
        ta = result.get("Ta")
        tu = result.get("Tu")
        if ta is not None and tu is not None:
            print(f"📊 Scores: Ta={ta:.2f} (approach), Tu={tu:.2f} (understanding)")
        
        print_ai_message(result)
        print_separator()
        
        turn += 1
        
        # If problem is solved but we haven't seen reflection yet, continue
        if result.get("solved", False) and result.get("current_state") != "REFLECTION":
            print("\n✨ Problem solved! Moving to reflection...")
            result = graph.invoke(None, config)
            print_separator()
            print_ai_message(result)
            print_separator()
            break
    
    # Print session summary
    print("\n📊 SESSION SUMMARY:")
    print(f"   Total turns: {turn}")
    print(f"   Final mode: {result.get('mode', 'N/A')}")
    print(f"   Problem solved: {result.get('solved', False)}")
    print(f"   Total messages: {len(result.get('messages', []))}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
