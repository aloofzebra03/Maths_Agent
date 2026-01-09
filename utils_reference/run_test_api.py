import os
import json
import time
import requests
from pprint import pprint
from typing import Optional, Dict, Any

from tester_agent.tester import TesterAgent
from tester_agent.evaluator import Evaluator
from tester_agent.personas import personas
from tester_agent.session_metrics import compute_and_upload_session_metrics
from tester_agent.simulation_descriptor import format_simulation_context_for_tester
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
if(os.getenv("POSTGRES_DATABASE_URL ")):
    print("✅ Loaded environment variables from .env file")

# # Set LANGCHAIN_API_KEY from LANGCHAIN_API_KEY if needed
# if not os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_API_KEY"):
#     os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

if os.getenv("LANGCHAIN_API_KEY"):
    print(f"✅ LangSmith tracing configured for project: {os.environ['LANGCHAIN_PROJECT']}")
    print(f"🔗 LangSmith endpoint: {os.environ['LANGCHAIN_ENDPOINT']}")
    
    # Test LangSmith connection
    try:
        from langsmith import Client
        client = Client()
        print("✅ LangSmith client connection successful")
    except Exception as e:
        print(f"❌ LangSmith connection test failed: {e}")
else:
    print("❌ Warning: LANGCHAIN_API_KEY not found. LangSmith tracing will not work.")


# ============================================================================
# API CLIENT CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000"  # Change this if your API is hosted elsewhere

class EducationalAgentAPIClient:
    """Client for interacting with the Educational Agent API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def start_session(self, concept_title: str, persona_name: str, session_label: Optional[str] = None, is_kannada: bool = False) -> Dict[str, Any]:
        url = f"{self.base_url}/session/start"
        payload = {
            "concept_title": concept_title,
            "persona_name": persona_name,
            "session_label": session_label,
            "is_kannada": is_kannada
        }
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def continue_session(self, thread_id: str, user_message: str) -> Dict[str, Any]:
        url = f"{self.base_url}/session/continue"
        payload = {
            "thread_id": thread_id,
            "user_message": user_message
        }
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_session_status(self, thread_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/session/status/{thread_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_session_history(self, thread_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/session/history/{thread_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_session_summary(self, thread_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/session/summary/{thread_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def delete_session(self, thread_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/session/{thread_id}"
        response = self.session.delete(url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_concepts(self) -> Dict[str, Any]:
        url = f"{self.base_url}/concepts"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def format_simulation_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    if not metadata.get("show_simulation", False):
        return None
    
    simulation_config = metadata.get("simulation_config", {})
    if not simulation_config:
        return None #If not simulation config, return None
    
    # Create a formatted simulation description similar to the agent's internal format
    sim_type = simulation_config.get("type", "unknown")
    parameters = simulation_config.get("parameters", {})
    agent_message = simulation_config.get("agent_message", "")
    
    description = f"Simulation Type: {sim_type}\n"
    description += f"Parameters: {json.dumps(parameters, indent=2)}\n"
    if agent_message:
        description += f"Agent Message: {agent_message}\n"
    
    return description


def run_test_api():
    
    # Check API health first
    print("\n" + "="*80)
    print("🏥 Checking API Health...")
    print("="*80)
    
    api_client = EducationalAgentAPIClient()
    
    try:
        health = api_client.health_check()
        print(f"✅ API Status: {health.get('status')}")
        print(f"📦 Agent Type: {health.get('agent_type')}")
        print(f"💾 Persistence: {health.get('persistence')}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Cannot connect to API at {API_BASE_URL}")
        print(f"   Make sure the API server is running with: uvicorn api_servers.api_server:app --host 0.0.0.0 --port 8000")
        print(f"   Error details: {e}")
        return
    
    # 1. Select Persona
    print("\n" + "="*80)
    print("👤 Select a persona to test:")
    print("="*80)
    for i, p in enumerate(personas):
        print(f"{i+1}. {p.name} - {p.description}")
    
    persona_idx = int(input("\nEnter persona number: ")) - 1
    persona = personas[persona_idx]
    
    print(f"\n✅ Selected persona: {persona.name}")
    
    # 1.5 Select Concept
    print("\n" + "="*80)
    print("📚 Select a concept to teach:")
    print("="*80)
    
    try:
        concepts_response = api_client.list_concepts()
        if concepts_response.get("success"):
            concepts = concepts_response.get("concepts", [])
            print(f"Found {len(concepts)} available concepts:\n")
            
            # Display concepts (already in title case from API)
            for i, concept in enumerate(concepts):
                print(f"{i+1}. {concept}")
            
            concept_choice = input("\nEnter concept number (or press Enter for default 'Pendulum And Its Time Period'): ").strip()
            
            if concept_choice and concept_choice.isdigit():
                concept_idx = int(concept_choice) - 1
                if 0 <= concept_idx < len(concepts):
                    # Use the concept from the list (already title cased)
                    concept_title = concepts[concept_idx]
                else:
                    print("Invalid choice, using default")
                    concept_title = "Pendulum And Its Time Period"
            else:
                concept_title = "Pendulum And Its Time Period"
        else:
            print("Could not retrieve concepts list, using default")
            concept_title = "Pendulum And Its Time Period"
    except Exception as e:
        print(f"Error retrieving concepts: {e}")
        print("Using default concept")
        concept_title = "Pendulum And Its Time Period"
    
    print(f"\n✅ Selected concept: {concept_title}")
    
    # 1.6 Select Language
    print("\n" + "="*80)
    print("🌐 Select language for the session:")
    print("="*80)
    print("1. English (default)")
    print("2. Kannada (ಕನ್ನಡ)")
    
    language_choice = input("\nEnter language number (default is 1): ").strip()
    is_kannada = (language_choice == "2")
    
    if is_kannada:
        print(f"\n✅ Selected language: Kannada (ಕನ್ನಡ)")
    else:
        print(f"\n✅ Selected language: English")
    
    # Get concept title (you can modify this to be dynamic if needed)
    # concept_title = "Pendulum and its Time Period"
    print(f"📚 Teaching concept: {concept_title}")
    
    # 2. Initialize Tester Agent (client-side only) with language preference
    tester_agent = TesterAgent(persona, is_kannada=is_kannada)
    
    # 3. Start Conversation via API
    print("\n" + "="*80)
    print("🚀 Starting Session via API...")
    print("="*80)
    
    try:
        # API server will automatically include language and concept in thread_id
        start_response = api_client.start_session(
            concept_title=concept_title,
            persona_name=persona.name,
            session_label=f"test-{persona.name.lower().replace(' ', '-')}",
            is_kannada=is_kannada
        )
    except requests.exceptions.RequestException as e:
        print(f"❌ Error starting session: {e}")
        return
    
    if not start_response.get("success"):
        print(f"❌ Failed to start session: {start_response.get('message')}")
        return
    
    thread_id = start_response.get("thread_id")
    session_id = start_response.get("session_id")
    agent_msg = start_response.get("agent_response")
    current_state = start_response.get("current_state")
    
    print(f"✅ Session started successfully!")
    print(f"   Thread ID: {thread_id}")
    print(f"   Session ID: {session_id}")
    print(f"   Current State: {current_state}")
    print(f"\n🤖 Educational Agent: {agent_msg}")
    
    # 4. Run Conversation Loop
    print("\n" + "="*80)
    print("💬 Starting Conversation Loop...")
    print("="*80)
    
    turn_count = 0
    # max_turns = 100  # Safety limit to prevent infinite loops
    
    while current_state != "END" :
        turn_count += 1
        print(f"\n--- Turn {turn_count} ---")
        
        # Check metadata for simulation
        metadata = start_response.get("metadata") if turn_count == 1 else continue_response.get("metadata")
        
        # Check if there's a simulation to describe
        simulation_description = None
        if metadata and metadata.get("show_simulation", False):
            simulation_description = format_simulation_from_metadata(metadata)
            
            if simulation_description:
                print("\n" + "="*80)
                print("🔬 SIMULATION DESCRIPTION FOR TESTER AGENT")
                print("="*80)
                print(simulation_description)
                print("="*80 + "\n")
        
        # Get tester response
        if simulation_description:
            # Get tester response with simulation context
            if hasattr(tester_agent, 'respond_with_simulation_context'):
                user_msg = tester_agent.respond_with_simulation_context(agent_msg, simulation_description)
            else:
                # Fallback: add context to agent message for older tester agent versions
                enhanced_agent_msg = f"{agent_msg}\n\n[SIMULATION CONTEXT: {simulation_description[:200]}...]"
                user_msg = tester_agent.respond(enhanced_agent_msg)
        else:
            # No simulation - normal response
            user_msg = tester_agent.respond(agent_msg)
        
        print(f"👤 Tester Agent ({persona.name}): {user_msg}")
        
        # Add delay to avoid overwhelming the API
        time.sleep(2)
        
        # Continue session via API
        try:
            continue_response = api_client.continue_session(thread_id, user_msg)
        except requests.exceptions.RequestException as e:
            print(f"❌ Error continuing session: {e}")
            break
        
        if not continue_response.get("success"):
            print(f"❌ Failed to continue session: {continue_response.get('message')}")
            break
        
        agent_msg = continue_response.get("agent_response")
        current_state = continue_response.get("current_state")
        
        print(f"🤖 Educational Agent: {agent_msg}")
        print(f"📍 Current State: {current_state}")
    
    # if turn_count >= max_turns:
    #     print(f"\n⚠️  Warning: Reached maximum turn limit ({max_turns})")
    
    # 5. Get Session Summary
    print("\n" + "="*80)
    print("📊 Retrieving Session Summary...")
    print("="*80)
    
    try:
        summary_response = api_client.get_session_summary(thread_id)
        
        if summary_response.get("success") and summary_response.get("exists"):
            session_summary = summary_response.get("summary", {})
            
            print("\n📋 Session Summary:")
            if session_summary:
                pprint(session_summary)
            else:
                print("⚠️  Session summary exists but is empty (session may not have reached END node yet)")
            
            # Save the session summary
            language_suffix = "kannada" if is_kannada else "english"
            summary_filename = f"session_summary_{session_id}_{language_suffix}_api.json"
            os.makedirs("test_reports", exist_ok=True)
            summary_path = os.path.join("test_reports", summary_filename)
            with open(summary_path, "w") as f:
                json.dump(session_summary, f, indent=2)
            print(f"\n✅ Session summary exported to {summary_path}")
        else:
            print("⚠️  No session summary available")
            session_summary = {}
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error retrieving session summary: {e}")
        session_summary = {}
    
    # 6. Get Session History for Evaluation
    print("\n" + "="*80)
    print("📜 Retrieving Session History...")
    print("="*80)
    
    try:
        history_response = api_client.get_session_history(thread_id)
        
        if history_response.get("success") and history_response.get("exists"):
            history_for_reports = history_response.get("messages", [])
            print(f"✅ Retrieved {len(history_for_reports)} messages from history")
        else:
            print("⚠️  No history available")
            history_for_reports = []
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error retrieving session history: {e}")
        history_for_reports = []
    
    # 7. Compute Session Metrics (if history is available)
    if history_for_reports:
        print("\n" + "="*80)
        print("📊 Computing Session Metrics...")
        print("="*80)
        
        try:
            # Get session state from status endpoint
            status_response = api_client.get_session_status(thread_id)
            session_state = status_response.get("progress", {}) if status_response.get("success") else {}
            
            session_metrics = compute_and_upload_session_metrics(
                session_id=session_id,
                history=history_for_reports,
                session_state=session_state,
                persona_name=persona.name
            )
            
            # Save metrics locally
            language_suffix = "kannada" if is_kannada else "english"
            metrics_filename = f"session_metrics_{session_id}_{language_suffix}_api.json"
            metrics_path = os.path.join("test_reports", metrics_filename)
            with open(metrics_path, "w") as f:
                json.dump(session_metrics.model_dump(), f, indent=2)
            print(f"✅ Session metrics saved to {metrics_path}")
        
        except Exception as e:
            print(f"❌ Error computing session metrics: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️  Continuing without metrics...")
            session_metrics = None
    else:
        session_metrics = None
    
    # 8. Evaluate Educational Quality (Pedagogical Assessment)
    if history_for_reports:
        print("\n" + "="*80)
        print("🎓 Evaluating Educational Quality...")
        print("="*80)
        
        evaluator = Evaluator()
        evaluation = evaluator.evaluate(persona, history_for_reports)
        print("\n--- Educational Quality Evaluation ---")
        print(evaluation)
        
        # Clean up evaluation response
        clean_str = evaluation.strip()
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]
        clean_str = clean_str.strip()
        
        # 9. Save Comprehensive Report
        report = {
            "persona": persona.model_dump(),
            "educational_evaluation": json.loads(clean_str),  # Pedagogical quality assessment
            "history": history_for_reports,
            "thread_id": thread_id,
            "session_id": session_id,
            "api_test": True  # Flag to indicate this was an API test
        }
        
        # Include quantitative metrics in the report if available
        if session_metrics:
            report["session_metrics"] = session_metrics.model_dump()
        
        language_suffix = "kannada" if is_kannada else "english"
        report_path = f"test_reports/evaluation_{session_id}_{language_suffix}_api.json"
        os.makedirs("test_reports", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Evaluation report saved to {report_path}")
    else:
        print("\n⚠️  Skipping evaluation - no history available")
    
    # 10. Final Summary
    print("\n" + "="*80)
    print("✅ TEST COMPLETED")
    print("="*80)
    print(f"Session ID: {session_id}")
    print(f"Thread ID: {thread_id}")
    print(f"Persona: {persona.name}")
    print(f"Total Turns: {turn_count}")
    print(f"Final State: {current_state}")
    print("="*80)
    
    # Optional: Ask if user wants to delete the session
    delete_choice = input("\n🗑️  Do you want to delete this session from the API? (y/n): ").strip().lower()
    if delete_choice == 'y':
        try:
            delete_response = api_client.delete_session(thread_id)
            if delete_response.get("success"):
                print(f"✅ Session deleted: {delete_response.get('message')}")
            else:
                print(f"⚠️  Session deletion failed: {delete_response.get('message')}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error deleting session: {e}")


if __name__ == "__main__":
    run_test_api()
