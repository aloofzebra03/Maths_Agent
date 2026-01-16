"""
Test script for Educational Agent API
Demonstrates complete usage flow
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"🔍 {title}")
    print("=" * 80)


def print_response(response_data):
    """Pretty print API response"""
    print(json.dumps(response_data, indent=2))


def test_health_check():
    """Test health endpoint"""
    print_section("1. HEALTH CHECK")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print_response(response.json())
    return response.status_code == 200


def test_start_session():
    """Test starting a new session"""
    print_section("2. START NEW SESSION")
    
    payload = {
        "concept_title": "Pendulum and its Time Period",
        "student_id": "test_student_123",
        "persona_name": "Curious Student",
        "session_label": "test-session"
    }
    
    print(f"Request: POST {BASE_URL}/session/start")
    print_response(payload)
    
    response = requests.post(f"{BASE_URL}/session/start", json=payload)
    print(f"\nStatus Code: {response.status_code}")
    data = response.json()
    print_response(data)
    
    if response.status_code == 200:
        print(f"\n✅ Session created successfully!")
        print(f"📝 Thread ID: {data['thread_id']}")
        print(f"💬 Agent says: {data['agent_response'][:100]}...")
        return data['thread_id']
    else:
        print(f"\n❌ Failed to create session")
        return None


def test_continue_session(thread_id, message):
    """Test continuing a session with user input"""
    print_section(f"3. CONTINUE SESSION - Message: '{message}'")
    
    payload = {
        "thread_id": thread_id,
        "user_message": message
    }
    
    print(f"Request: POST {BASE_URL}/session/continue")
    print_response(payload)
    
    response = requests.post(f"{BASE_URL}/session/continue", json=payload)
    print(f"\nStatus Code: {response.status_code}")
    data = response.json()
    print_response(data)
    
    if response.status_code == 200:
        print(f"\n✅ Response received!")
        print(f"🎯 Current State: {data['current_state']}")
        print(f"💬 Agent says: {data['agent_response'][:150]}...")
        
        # Check for simulation
        metadata = data.get('metadata', {})
        if metadata.get('show_simulation'):
            print(f"\n⚠️  SIMULATION TRIGGER DETECTED!")
            print(f"📊 Simulation Config:")
            print_response(metadata.get('simulation_config', {}))
        
        # Check for images
        if metadata.get('enhanced_message_metadata'):
            print(f"\n🖼️  IMAGE AVAILABLE!")
            image_data = metadata['enhanced_message_metadata'].get('image', {})
            print(f"URL: {image_data.get('url')}")
            print(f"Caption: {image_data.get('caption')}")
        
        return data
    else:
        print(f"\n❌ Failed to continue session")
        return None


def test_get_status(thread_id):
    """Test getting session status"""
    print_section("4. GET SESSION STATUS")
    
    response = requests.get(f"{BASE_URL}/session/status/{thread_id}")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print_response(data)
    
    if response.status_code == 200 and data['exists']:
        progress = data['progress']
        print(f"\n✅ Session Status Retrieved!")
        print(f"🎯 Current State: {data['current_state']}")
        print(f"📊 Concepts: {progress.get('concepts', [])}")
        print(f"📍 Current Concept Index: {progress.get('current_concept_idx', 0)}")
        print(f"🔬 In Simulation: {progress.get('in_simulation', False)}")


def test_get_history(thread_id):
    """Test getting session history"""
    print_section("5. GET SESSION HISTORY")
    
    response = requests.get(f"{BASE_URL}/session/history/{thread_id}")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    
    if response.status_code == 200 and data['exists']:
        messages = data['messages']
        transitions = data['node_transitions']
        
        print(f"\n✅ History Retrieved!")
        print(f"📨 Total Messages: {len(messages)}")
        print(f"🔄 Node Transitions: {len(transitions)}")
        
        print(f"\n📜 Recent Messages (last 3):")
        for msg in messages[-3:]:
            role = msg['role']
            content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
            node = msg.get('node', 'N/A')
            print(f"  [{role.upper()}] ({node}): {content}")


def test_get_summary(thread_id):
    """Test getting session summary"""
    print_section("6. GET SESSION SUMMARY")
    
    response = requests.get(f"{BASE_URL}/session/summary/{thread_id}")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print_response(data)
    
    if response.status_code == 200 and data['exists']:
        print(f"\n✅ Summary Retrieved!")
        if data.get('quiz_score') is not None:
            print(f"📊 Quiz Score: {data['quiz_score']}/100")
        if data.get('misconception_detected') is not None:
            print(f"🔍 Misconception Detected: {data['misconception_detected']}")


def test_list_sessions():
    """Test listing all active sessions"""
    print_section("7. LIST ALL SESSIONS")
    
    response = requests.get(f"{BASE_URL}/sessions")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print_response(data)
    
    if response.status_code == 200:
        print(f"\n✅ {data['total_sessions']} active session(s)")


def test_delete_session(thread_id):
    """Test deleting a session"""
    print_section("8. DELETE SESSION")
    
    response = requests.delete(f"{BASE_URL}/session/{thread_id}")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print_response(data)
    
    if response.status_code == 200:
        print(f"\n✅ Session deleted!")


def run_full_test():
    """Run complete test suite"""
    print("\n" + "=" * 80)
    print("🧪 EDUCATIONAL AGENT API - FULL TEST SUITE")
    print("=" * 80)
    
    # Test health
    if not test_health_check():
        print("\n❌ Server is not healthy. Exiting...")
        return
    
    time.sleep(1)
    
    # Start session
    thread_id = test_start_session()
    if not thread_id:
        print("\n❌ Could not start session. Exiting...")
        return
    
    time.sleep(1)
    
    # Continue with a few interactions
    messages = [
        "Yes, I'm ready to learn!",
        "A pendulum is something that swings back and forth",
        "I think it depends on the length",
    ]
    
    for i, msg in enumerate(messages, 1):
        time.sleep(1)
        result = test_continue_session(thread_id, msg)
        if not result:
            break
        
        # Stop after a few interactions for testing
        if i >= 3:
            break
    
    time.sleep(1)
    
    # Get status
    test_get_status(thread_id)
    
    time.sleep(1)
    
    # Get history
    test_get_history(thread_id)
    
    time.sleep(1)
    
    # Get summary
    test_get_summary(thread_id)
    
    time.sleep(1)
    
    # List sessions
    test_list_sessions()
    
    time.sleep(1)
    
    # Delete session (optional - comment out to keep session alive)
    # test_delete_session(thread_id)
    
    print("\n" + "=" * 80)
    print("✅ TEST SUITE COMPLETED!")
    print("=" * 80)
    print(f"\n💡 TIP: Your session is still active with thread_id: {thread_id}")
    print(f"You can continue interacting with it using the API or delete it later.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        run_full_test()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API server.")
        print(f"Please ensure the server is running on {BASE_URL}")
        print("Start it with: python api_servers/api_server.py")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
