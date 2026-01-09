from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage

from langgraph.types import Command

# Import the graph factory that returns a compiled graph WITHOUT callbacks baked in
from educational_agent_optimized_langsmith.graph import build_graph


class EducationalAgent:
    def __init__(
        self,
        session_label: Optional[str] = None,
        user_id: Optional[str] = None,
        persona_name: Optional[str] = None,
    ):
        
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Prefer explicit session label; otherwise use persona; otherwise "tester"
        base = session_label or persona_name or "tester"
        self.base = base
        self.persona_name = persona_name

        # Public identifiers (handy in LangSmith UI)
        self.session_id = f"{base}-{ts}"
        self.thread_id = f"{base}-thread-{ts}"
        self.user_id = user_id or "local-tester"

        # Metadata for the parent run in LangSmith
        tags = [self.base, "educational-agent"]
        if self.persona_name:
            tags.append(f"persona:{self.persona_name}")

        self._metadata: Dict[str, Any] = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tags": tags,
            "run_started_at": ts,
        }

        base_graph = build_graph()

        # Configure the graph with metadata for LangSmith
        self.graph = base_graph.with_config({
            "metadata": self._metadata,
            "tags": tags,
        })

        print("LangSmith header attached")

        # Optional local state slot if you need to stash anything custom
        self.state: Dict[str, Any] = {}

    def start(self) -> str:
        final_text = ""
        last_state: Dict[str, Any] = {}
        # events = self.graph.stream(
        #     {"messages": [HumanMessage(content="__start__")],
        #      "history": self.state.get("history", [])},  # seed for Gemini
        #     stream_mode="values",
        #     config={"configurable": {"thread_id": self.thread_id}},
        # )
        # for state in events:
        #     if isinstance(state, dict):
        #         last_state = state
        #         if state.get("agent_output"):
        #             final_text = state["agent_output"]
        
        result = self.graph.invoke(
            {"messages": [HumanMessage(content="__start__")]},
            config={"configurable": {"thread_id": self.thread_id}},
        )
        if isinstance(result, dict):
            last_state = result
            if result.get("agent_output"):
                final_text = result["agent_output"]

        if last_state:
            self.state = last_state

        return final_text
    
    def post(self, user_text: str) -> str:
        final_text = ""
        last_state: Dict[str, Any] = {}
        
        # Create user message
        user_message = HumanMessage(content=user_text)
        
        cmd = Command(
            resume=True,
            update={
                "messages": [user_message],  # LangGraph will add this to existing messages
            },
        )

        # events = self.graph.stream(
        #     cmd,
        #     stream_mode="values",
        #     config={"configurable": {"thread_id": self.thread_id}},
        # )
        # for state in events:
        #     if isinstance(state, dict):
        #         last_state = state
        #         if state.get("agent_output"):
        #             final_text = state["agent_output"]
        #         else:
        #             try:
        #                 msgs = state.get("messages") or []
        #                 if msgs and getattr(msgs[-1], "type", None) == "ai":
        #                     final_text = getattr(msgs[-1], "content", final_text) or final_text
        #             except Exception:
        #                 pass

        result = self.graph.invoke(
            cmd,
            config={"configurable": {"thread_id": self.thread_id}},
        )
        
        # Debug logging: Verify message state
        if isinstance(result, dict):
            last_state = result
            messages = result.get("messages", [])
            
            print(f"🔍 AGENT DEBUG - Post-invoke state:")
            print(f"📊 Messages count: {len(messages)}")
            
            # Show last few messages for verification
            if messages:
                print("📜 Last 3 messages:")
                for i, msg in enumerate(messages[-3:]):
                    msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                    content = (msg.content[:50] + "...") if len(msg.content) > 50 else msg.content
                    print(f"  {len(messages)-3+i+1}. {msg_type}: {content}")
            
            if result.get("agent_output"):
                final_text = result["agent_output"]
            else:
                try:
                    msgs = result.get("messages") or []
                    if msgs and getattr(msgs[-1], "type", None) == "ai":
                        final_text = getattr(msgs[-1], "content", final_text) or final_text
                except Exception:
                    pass

        # <-- persist full state
        if last_state:
            self.state = last_state

        return final_text
    
    def current_state(self) -> str:
        return self.state.get("current_state", "")

    def get_history_for_reports(self) -> List[Dict[str, Any]]:
        """Convert messages to history format for reports and metrics"""
        history = []
        messages = self.state.get("messages", [])
        current_node = "unknown"
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # Skip the initial "__start__" message
                if msg.content != "__start__":
                    history.append({
                        "role": "user",
                        "content": msg.content
                    })
            elif isinstance(msg, AIMessage):
                # Try to get the current node from agent state, fallback to tracking from messages
                current_node = self.state.get("current_state", current_node)
                history.append({
                    "role": "assistant", 
                    "content": msg.content,
                    "node": current_node
                })
        
        return history


    def session_info(self) -> Dict[str, str]:
        return {
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "tags": ", ".join(self._metadata.get("tags", [])),
        }
