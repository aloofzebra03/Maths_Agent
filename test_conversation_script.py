import os
import sys
from langchain_core.messages import HumanMessage
from educational_agent_math_tutor.graph import graph
from langgraph.types import Command

def test_flow():
    config = {"configurable": {"thread_id": "test_interaction_1"}}
    
    # 1. Start session
    print("\n--- START ---")
    initial_state = {
        "problem_id": "integer_operations_1",
        "messages": [HumanMessage(content="start")],
        "is_kannada": False
    }
    result = graph.invoke(initial_state, config)
    print(f"Agent ({result.get('current_state')}): {result.get('agent_output')}")
    
    # helper for subsequent turns
    def turn(user_input):
        print(f"\n--- User says: '{user_input}' ---")
        cmd = Command(
            resume=True,
            update={"messages": [HumanMessage(content=user_input)], "is_kannada": False}
        )
        res = graph.invoke(cmd, config)
        print(f"Agent ({res.get('current_state')}): {res.get('agent_output')}")
        return res

    # 2. Answer "I don't know" to assessment
    turn("I don't know")
    turn("sure")
    # This should trigger CONCEPT mode.
    # We should get asked a micro-check question. Let's see what it asks!

if __name__ == "__main__":
    test_flow()
