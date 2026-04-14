"""
Automated Test Runner for LangGraph-based Math Tutoring Agent

Integrates:
- educational_agent_math_tutor (LangGraph-based agent)
- tester_agent (simulated student personas)
- problems_json (problem database)

Features:
- Interactive problem selection from all JSON files
- Persona-based automated testing
- Session metrics and evaluation reports
- LangSmith integration (optional)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from pprint import pprint

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from dotenv import load_dotenv

# Import agent components
from educational_agent_math_tutor import graph, MathAgentState
from tester_agent.tester import TesterAgent
from tester_agent.evaluator import Evaluator
from tester_agent.personas import personas
from tester_agent.session_metrics import compute_and_upload_session_metrics

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# LangSmith configuration check
if os.getenv("LANGCHAIN_API_KEY"):
    print(f"✅ LangSmith tracing configured for project: {os.environ.get('LANGCHAIN_PROJECT', 'N/A')}")
    print(f"🔗 LangSmith endpoint: {os.environ.get('LANGCHAIN_ENDPOINT', 'N/A')}")
    
    try:
        from langsmith import Client
        client = Client()
        print("✅ LangSmith client connection successful")
    except Exception as e:
        print(f"⚠️  LangSmith connection test failed: {e}")
else:
    print("⚠️  Warning: LANGCHAIN_API_KEY not found. LangSmith tracing will not work.")


# ============================================================================
# Problem Loading (inspired by streamlit_ui/app.py)
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
        print(f"❌ Error: Problems directory not found: {problems_path}")
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
                        'difficulty': problem_data.get('difficulty', 'unknown'),
                        'file': filepath.name
                    }
                    
        except Exception as e:
            print(f"⚠️  Warning: Error reading {filepath.name}: {e}")
            continue
    
    return problems


# ============================================================================
# History Conversion for Reports
# ============================================================================

def convert_messages_to_history(messages) -> List[Dict]:
    """
    Convert LangGraph messages to history format for reports.
    
    Args:
        messages: List of LangGraph message objects
        
    Returns:
        List of dicts with {role: "ai"/"human", content: "..."}
    """
    history = []
    
    for msg in messages:
        msg_type = msg.__class__.__name__
        
        if msg_type == "HumanMessage":
            role = "human"
        elif msg_type == "AIMessage":
            role = "ai"
        else:
            continue  # Skip system messages or other types
        
        history.append({
            "role": role,
            "content": msg.content
        })
    
    return history


# ============================================================================
# Main Test Runner
# ============================================================================

def run_test():
    """Run automated test session with persona-based tester agent."""
    
    print("\n" + "="*80)
    print("🎓 MATH TUTORING AGENT - AUTOMATED TEST SESSION")
    print("="*80)
    
    # ========================================================================
    # Step 1: Load all problems
    # ========================================================================
    
    print("\n📚 Loading problems from problems_json directory...")
    all_problems = load_all_problems()
    
    if not all_problems:
        print("❌ Error: No problems found!")
        return
    
    print(f"✅ Loaded {len(all_problems)} problems from problems_json/")
    
    # ========================================================================
    # Step 2: Problem Selection
    # ========================================================================
    
    print("\n" + "="*80)
    print("📚 Select a problem to test:")
    print("="*80)
    
    # Sort problems by ID for consistent ordering
    problem_ids = sorted(all_problems.keys())
    
    # Display problems with details
    for i, problem_id in enumerate(problem_ids, 1):
        problem = all_problems[problem_id]
        topic = problem['topic']
        difficulty = problem['difficulty']
        question = problem['question']
        
        # Truncate long questions
        if len(question) > 60:
            question = question[:60] + "..."
        
        print(f"{i}. {problem_id}")
        print(f"   [{topic}] ({difficulty})")
        print(f"   Question: {question}")
        print()
    
    # Get user selection
    while True:
        try:
            choice = input("Enter problem number: ").strip()
            problem_idx = int(choice) - 1
            
            if 0 <= problem_idx < len(problem_ids):
                selected_problem_id = problem_ids[problem_idx]
                selected_problem = all_problems[selected_problem_id]
                break
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(problem_ids)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n❌ Test cancelled by user")
            return
    
    print(f"\n✅ Selected problem: {selected_problem_id}")
    print(f"   Topic: {selected_problem['topic']}")
    print(f"   Difficulty: {selected_problem['difficulty']}")
    print(f"   Question: {selected_problem['question']}")
    
    # ========================================================================
    # Step 3: Persona Selection
    # ========================================================================
    
    print("\n" + "="*80)
    print("👤 Select a persona to test:")
    print("="*80)
    
    for i, p in enumerate(personas, 1):
        print(f"{i}. {p.name}")
        print(f"   {p.description}")
        print()
    
    # Get user selection
    while True:
        try:
            choice = input("Enter persona number: ").strip()
            persona_idx = int(choice) - 1
            
            if 0 <= persona_idx < len(personas):
                selected_persona = personas[persona_idx]
                break
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(personas)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n❌ Test cancelled by user")
            return
    
    print(f"\n✅ Selected persona: {selected_persona.name}")

    # ========================================================================
    # Step 3b: Language Selection
    # ========================================================================

    print("\n" + "="*80)
    print("🌐 Select language:")
    print("="*80)
    print("1. English")
    print("2. Kannada (ಕನ್ನಡ)")
    print()

    while True:
        try:
            lang_choice = input("Enter language number (default 1): ").strip() or "1"
            if lang_choice in ("1", "2"):
                break
            print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n❌ Test cancelled by user")
            return

    is_kannada = lang_choice == "2"
    language = "kannada" if is_kannada else "english"
    print(f"\n✅ Selected language: {language.capitalize()}")

    # ========================================================================
    # Step 4: Session Initialization
    # ========================================================================

    # Create session ID (matching format from run_test_api.py)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_name_slug = selected_persona.name.lower().replace(' ', '-')
    session_id = f"gemma4(26b)_test_{selected_problem_id}_{persona_name_slug}_{language}_{timestamp}"
    thread_id = session_id

    print("\n" + "="*80)
    print("🚀 Starting Session")
    print("="*80)
    print(f"Session ID: {session_id}")
    print(f"Thread ID: {thread_id}")
    print()

    # Initialize tester agent
    tester_agent = TesterAgent(selected_persona, is_kannada=is_kannada)
    
    # LangGraph config
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # ========================================================================
    # Step 5: Start the Graph (Initial Invoke)
    # ========================================================================
    
    print("▶️  Invoking START node...")
    
    initial_state = {
        "problem_id": selected_problem_id,
        "messages": [HumanMessage(content="Hello")],
        "is_kannada": is_kannada,
    }
    
    try:
        result = graph.invoke(initial_state, config)
    except Exception as e:
        print(f"❌ Error during initial graph invoke: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Extract agent's initial greeting
    agent_msg = result.get("agent_output", "No response from agent")
    current_state = result.get("current_state", "UNKNOWN")
    
    # ========================================================================
    # Step 6: Conversation Loop
    # ========================================================================
    
    print("\n" + "="*80)
    print("💬 CONVERSATION")
    print("="*80)
    
    turn_count = 0
    
    while current_state != "END":
        turn_count += 1
        print(f"\n--- Turn {turn_count} ---")
        
        # Display agent message
        print(f"🤖 Agent: {agent_msg}")
        
        # Display scores if available
        if result.get("mode"):
            mode = result.get("mode", "N/A")
            ta_score = result.get("Ta", 0.0)
            tu_score = result.get("Tu", 0.0)
            print(f"📊 Mode: {mode} | Ta: {ta_score:.2f} | Tu: {tu_score:.2f}")
        
        # Get tester agent response
        try:
            user_msg = tester_agent.respond(agent_msg)
        except Exception as e:
            print(f"❌ Error getting tester response: {e}")
            break
        
        print(f"👨‍🎓 Tester ({selected_persona.name}): {user_msg}")
        
        # Small delay to avoid overwhelming the LLM API
        time.sleep(2) 
        
        cmd = Command(resume=True,
                        update={
                            "messages": [HumanMessage(content=user_msg)],  # LangGraph will add this to existing messages
                        },
                    )

        # Resume graph with student response
        try:
            result = graph.invoke(
                cmd,
                config
            )
        except Exception as e:
            print(f"❌ Error during graph continuation: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Update state
        agent_msg = result.get("agent_output", "No response from agent")
        current_state = result.get("current_state", "UNKNOWN")
    
    # ========================================================================
    # Step 7: Session Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("✅ SESSION COMPLETE")
    print("="*80)
    print(f"Total Turns: {turn_count}")
    print(f"Final State: {current_state}")
    
    # Get session summary from final state
    session_summary = result.get("session_summary", {})
    
    if session_summary:
        print("\n📊 Session Summary:")
        pprint(session_summary)
    
    # ========================================================================
    # Step 8: Save Session Summary
    # ========================================================================
    
    os.makedirs("test_reports", exist_ok=True)
    
    summary_filename = f"session_summary_{session_id}.json"
    summary_path = os.path.join("test_reports", summary_filename)
    
    with open(summary_path, "w") as f:
        json.dump(session_summary, f, indent=2)
    
    print(f"\n💾 Session summary saved to: {summary_path}")
    
    # ========================================================================
    # Step 9: Compute Session Metrics (if LangSmith available)
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 Computing Session Metrics...")
    print("="*80)
    
    # Convert messages to history format
    messages = result.get("messages", [])
    history_for_reports = convert_messages_to_history(messages)
    
    session_metrics = None
    
    try:
        session_metrics = compute_and_upload_session_metrics(
            session_id=session_id,
            history=history_for_reports,
            session_state=result,
            persona_name=selected_persona.name
        )
        
        # Save metrics locally
        metrics_filename = f"session_metrics_{session_id}.json"
        metrics_path = os.path.join("test_reports", metrics_filename)
        
        with open(metrics_path, "w") as f:
            json.dump(session_metrics.model_dump(), f, indent=2)
        
        print(f"✅ Session metrics saved to: {metrics_path}")
    
    except Exception as e:
        print(f"⚠️  Error computing session metrics: {e}")
        print("   Continuing without metrics...")
    
    # ========================================================================
    # Step 10: Evaluate Educational Quality
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎓 Evaluating Educational Quality...")
    print("="*80)
    
    evaluator = Evaluator()
    
    try:
        evaluation = evaluator.evaluate(selected_persona, history_for_reports)
        
        print("\n--- Educational Quality Evaluation ---")
        print(evaluation)
        
        # Clean up evaluation response (remove markdown code blocks if present)
        clean_str = evaluation.strip()
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]
        clean_str = clean_str.strip()
        
        evaluation_data = json.loads(clean_str)
    
    except Exception as e:
        print(f"⚠️  Error during evaluation: {e}")
        evaluation_data = {"error": str(e)}
    
    # ========================================================================
    # Step 11: Save Comprehensive Report
    # ========================================================================
    
    print("\n" + "="*80)
    print("💾 Saving Comprehensive Report...")
    print("="*80)
    
    report = {
        "session_id": session_id,
        "problem_id": selected_problem_id,
        "problem_details": selected_problem,
        "persona": selected_persona.model_dump(),
        "total_turns": turn_count,
        "session_summary": session_summary,
        "educational_evaluation": evaluation_data,
        "history": history_for_reports,
    }
    
    # Include metrics if available
    if session_metrics:
        report["session_metrics"] = session_metrics.model_dump()
    
    report_filename = f"evaluation_{session_id}.json"
    report_path = os.path.join("test_reports", report_filename)
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Comprehensive report saved to: {report_path}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎉 TEST COMPLETE")
    print("="*80)
    print(f"Session ID: {session_id}")
    print(f"Problem: {selected_problem_id}")
    print(f"Persona: {selected_persona.name}")
    print(f"Total Turns: {turn_count}")
    print(f"Reports saved to: test_reports/")
    print("="*80)


if __name__ == "__main__":
    try:
        run_test()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
