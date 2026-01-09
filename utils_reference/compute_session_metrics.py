#!/usr/bin/env python3
"""
Standalone script to compute and upload session metrics for any educational session.

Usage:
    python compute_session_metrics.py --session_id <session_id> --history_file <path_to_history.json>
    
Or use programmatically:
    from compute_session_metrics import compute_metrics_from_file
    metrics = compute_metrics_from_file("session_history.json", "my_session_id")
"""

import argparse
import json
import sys
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from tester_agent.session_metrics import compute_and_upload_session_metrics, SessionMetrics

# Load environment variables
load_dotenv()


def load_session_data(file_path: str) -> Dict[str, Any]:
    """Load session data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error: Failed to load session data from {file_path}: {e}")
        raise RuntimeError(f"Failed to load session data: {e}") from e


def extract_history_and_state(data: Dict[str, Any]) -> tuple[List[Dict], Dict[str, Any]]:
    """Extract history and state from loaded session data"""
    
    # Try different possible data structures
    if "history" in data and "session_state" in data:
        # Direct format
        return data["history"], data.get("session_state", {})
    
    elif "history" in data:
        # History only format (common in reports)
        return data["history"], {}
    
    elif isinstance(data, list):
        # Raw history list
        return data, {}
    
    elif "educational_evaluation" in data and "history" in data:
        # Updated evaluation report format (post-refactor)
        return data["history"], {}
    
    elif "evaluation" in data and "history" in data:
        # Legacy evaluation report format (pre-refactor)
        return data["history"], {}
    
    else:
        print("❌ Error: Could not extract history from data. Expected formats:")
        print("   - {'history': [...], 'session_state': {...}}")
        print("   - {'history': [...]}")
        print("   - [...]  (raw history list)")
        print("   - {'educational_evaluation': {...}, 'history': [...]}  (updated evaluation report)")
        print("   - {'evaluation': {...}, 'history': [...]}  (legacy evaluation report)")
        raise RuntimeError("Could not extract history from data - unsupported format")


def compute_metrics_from_file(file_path: str, 
                            session_id: str = None, 
                            persona_name: str = None,
                            upload: bool = True) -> SessionMetrics:
    """
    Compute session metrics from a file containing session data
    
    Args:
        file_path: Path to JSON file with session data
        session_id: Session ID (if not provided, will be extracted or generated)
        persona_name: User persona name
        upload: Whether to upload to Langfuse
        
    Returns:
        SessionMetrics object
    """
    
    # Load and parse data
    data = load_session_data(file_path)
    history, session_state = extract_history_and_state(data)
    
    # Extract session_id if not provided
    if not session_id:
        session_id = (
            data.get("session_id") or 
            session_state.get("session_id") or
            data.get("persona", {}).get("name", "unknown") + "_session"
        )
    
    # Extract persona if not provided
    if not persona_name:
        persona_name = (
            data.get("persona", {}).get("name") or
            session_state.get("persona_name") or
            "unknown_persona"
        )
    
    print(f"📊 Computing metrics for session: {session_id}")
    print(f"👤 Persona: {persona_name}")
    print(f"💬 History length: {len(history)} interactions")
    
    # Compute metrics
    try:
        if upload:
            metrics = compute_and_upload_session_metrics(
                session_id=session_id,
                history=history,
                session_state=session_state,
                persona_name=persona_name
            )
        else:
            from tester_agent.session_metrics import MetricsComputer
            computer = MetricsComputer()
            metrics = computer.compute_metrics(session_id, history, session_state, persona_name)
            print("📊 Metrics computed (not uploaded)")
        
        return metrics
    
    except Exception as e:
        print(f"❌ Error: Failed to compute metrics: {e}")
        raise RuntimeError(f"Failed to compute metrics: {e}") from e


def main():
    """Command line interface for computing session metrics"""
    parser = argparse.ArgumentParser(
        description="Compute and upload session metrics for educational sessions"
    )
    parser.add_argument(
        "history_file", 
        help="Path to JSON file containing session history/data"
    )
    parser.add_argument(
        "--session_id", 
        help="Session ID (if not provided, will be extracted from data)"
    )
    parser.add_argument(
        "--persona", 
        help="User persona name (if not provided, will be extracted from data)"
    )
    parser.add_argument(
        "--no-upload", 
        action="store_true",
        help="Compute metrics without uploading to Langfuse"
    )
    parser.add_argument(
        "--output", 
        help="Save computed metrics to this file"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.history_file).exists():
        print(f"❌ File not found: {args.history_file}")
        sys.exit(1)
    
    # Compute metrics
    try:
        metrics = compute_metrics_from_file(
            file_path=args.history_file,
            session_id=args.session_id,
            persona_name=args.persona,
            upload=not args.no_upload
        )
        
        # Display results
        print("\n" + "="*50)
        print("📊 SESSION METRICS SUMMARY")
        print("="*50)
        print(f"Session ID: {metrics.session_id}")
        print(f"User Type: {metrics.user_type}")
        print(f"Concepts Covered: {metrics.num_concepts_covered}")
        print(f"Quiz Score: {metrics.quiz_score:.1f}%")
        print(f"Engagement Rating: {metrics.user_engagement_rating:.1f}/5")
        print(f"Interest Rating: {metrics.user_interest_rating:.1f}/5")
        print(f"Enjoyment Probability: {metrics.enjoyment_probability:.0%}")
        print(f"Adaptability: {'Yes' if metrics.adaptability else 'No'}")
        print(f"Error Handling Count: {metrics.error_handling_count}")
        print(f"Total Interactions: {metrics.total_interactions}")
        print(f"Session Duration: {metrics.session_duration:.1f} minutes")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(metrics.model_dump(), f, indent=2)
            print(f"\n💾 Metrics saved to: {args.output}")
        
        print("\n✅ Metrics computation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
