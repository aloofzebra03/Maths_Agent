"""
API Tracker Utils - Plug-and-Play Quota Management

A standalone, thread-safe API quota tracker designed to drop into any
LangGraph/LangChain agent with zero configuration.

🎯 Core Features:
- Thread-safe tracking using atomic list operations (no locks!)
- Per-minute and per-day rate limiting with sliding windows
- Intelligent API-model selection (least-used + random from top-3)
- Email notifications when quotas exhausted
- Typed exceptions for clean error handling

📦 Quick Start:
    from api_tracker_utils import (
        track_model_call,
        get_next_available_api_model_pair,
        get_tracker_stats,
    )
    
    # Get available API-model pair (handles rate limits automatically)
    api_key, model = get_next_available_api_model_pair()
    
    # Track the API call (before making it)
    track_model_call(api_key, model)
    
    # Make your LLM call
    response = llm.invoke(...)

🔧 Setup:
1. Set environment variables: GOOGLE_API_KEY_1 through GOOGLE_API_KEY_7
2. Configure rate limits in config.py (or use defaults)
3. Import and use - that's it!

⚡ Thread-Safe: Uses Python's atomic list.append() - works perfectly
   with uvicorn's 40-thread pool (single worker) with zero locks.
"""
