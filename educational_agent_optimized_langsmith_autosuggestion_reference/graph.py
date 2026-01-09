from langchain_core.runnables import RunnableConfig
import uuid
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import os
import dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# from langfuse import get_client
# from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool


from educational_agent_optimized_langsmith_autosuggestion.main_nodes_simulation_agent_no_mh import (
    start_node, apk_node, ci_node, ge_node,
    ar_node, tc_node, rlc_node, end_node,
    autosuggestion_manager_node,
)

# ▶ NEW: import simulation agent nodes
from educational_agent_optimized_langsmith_autosuggestion.simulation_nodes_no_mh_ge import (
    sim_concept_creator_node,
    sim_vars_node,
    sim_action_node,
    sim_expect_node,
    sim_execute_node,
    sim_observe_node,
    sim_insight_node,
    sim_reflection_node,
)

dotenv.load_dotenv(dotenv_path=".env", override=True)

# -----------------------------------------------------------------------------
# // 3. Define AgentState TypedDict
# -----------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    current_state: str
    last_user_msg: str
    agent_output: str
    asked_apk: bool
    asked_ci: bool
    asked_ge: bool
    asked_mh: bool
    asked_ar: bool
    asked_tc: bool
    asked_rlc: bool
    apk_tries: int
    ci_tries: int
    ge_tries: int
    mh_tries: int
    rlc_tries: int
    definition_echoed: bool
    sim_concepts: List[str]
    sim_total_concepts: int
    sim_current_idx: int
    concepts_completed: bool
    in_simulation: bool
    misconception_detected: bool
    retrieval_score: float
    transfer_success: bool
    last_correction: str
    quiz_score: float
    session_summary: Dict[str, Any]
    # NEW: Simulation-related state fields
    sim_variables: List[Dict[str, Any]]  # List of variable dictionaries for JSON serialization
    sim_action_config: Dict[str, Any]
    show_simulation: bool
    simulation_config: Dict[str, Any]
    # NEW: Memory optimization fields
    node_transitions: List[Dict[str, Any]]
    summary: str
    summary_last_index: int
    enhanced_message_metadata: Dict[str, Any]
    # NEW: Language preference
    is_kannada: bool
    # NEW: Concept title
    concept_title: str
    # Model selection - accepts updates, uses last non-None value
    # model: Annotated[str, lambda old, new: new if new is not None else old]
    # NEW: Restructured autosuggestion fields (4 distinct types)
    autosuggestions: List[str]  # Final combined suggestions to display [positive, negative, special, dynamic]
    positive_autosuggestion: str  # Selected positive/affirmative suggestion
    negative_autosuggestion: str  # Selected negative/uncertain suggestion
    special_handling_autosuggestion: str  # Selected special handling suggestion (triggers handlers)
    dynamic_autosuggestion: str  # Generated exploratory suggestion
    clicked_autosuggestion: Annotated[bool, lambda x, y: y if y is not None else x]  # True if user clicked autosuggestion button, False if typed
    handler_triggered: bool  # True if handler was triggered by autosuggestion manager
    student_level: Annotated[str, lambda x, y: y if y is not None else x]  # Student ability level: "low", "medium", or "advanced"

def _wrap(fn):
    def inner(state: AgentState) -> AgentState:
        print(f"🔧 _WRAP DEBUG - Node processing started")
        print(f"📊 Messages count: {len(state.get('messages', []))}")
        
        # CAPTURE OLD STATE BEFORE PROCESSING
        old_state = state.get("current_state")
        
        msgs = state.get("messages", [])
        if msgs and isinstance(msgs[-1], HumanMessage):
            print(f"📝 Last message is HumanMessage")
            text = msgs[-1].content or ""
            print(msgs)
            print(msgs[-1].content)
            print(state.get("last_user_msg"))
            if text and text != state.get("last_user_msg"):
                print(f"📝 Detected new user message: {text}...")
                state["last_user_msg"] = text
                print(f"📝 Updated last_user_msg: {text[:50]}...")
            else :
                print(f"📝 No change in last_user_msg")
                # raise ValueError("No new user message detected in the last HumanMessage")
        
        # CALL THE ORIGINAL NODE FUNCTION
        result = fn(state)
        
        # Handle both full state returns (legacy) and partial state updates (LangGraph best practice)
        if isinstance(result, dict) and result.get("messages") is None:
            # Partial state update - merge with existing state (LangGraph best practice)
            print(f"🔄 _WRAP DEBUG - Merging partial state update with keys: {list(result.keys())}")
            state.update(result)
        else:
            # Full state return (legacy behavior) - update the original state dictionary
            state.update(result)
        
        st = state # Always use the original state reference
        
        # CAPTURE NEW STATE AFTER PROCESSING
        new_state = st.get("current_state")

        print(st.get("messages"))
        final_message_count = len(st.get("messages", []))
        
        # TRACK TRANSITION IF STATE CHANGED
        # The transition happens AFTER the current agent response is added
        if old_state != new_state and old_state is not None:
            transitions = st.setdefault("node_transitions", [])
            transitions.append({
                "from_node": old_state,
                "to_node": new_state,
                "transition_after_message_index": final_message_count,
            })
            print(f"🔄 NODE TRANSITION: {old_state} -> {new_state} after message {final_message_count}")
        
        # UPDATE SESSION SUMMARY (continuous throughout session)
        if "session_summary" in st:
            st["session_summary"].update({
                "quiz_score": st.get("retrieval_score"),
                "transfer_success": st.get("transfer_success", False),
                "definition_echoed": st.get("definition_echoed", False),
                "misconception_detected": st.get("misconception_detected", False),
                "last_user_msg": st.get("last_user_msg", ""),
                "current_state": new_state,
            })
        
        print(f"🏁 _WRAP DEBUG - Node processing completed")
        print(f"📊 Final messages count: {final_message_count}")
        return st
    return inner

# Node wrappers (core)
def _START(s): return _wrap(start_node)(s)
def _APK(s):   return _wrap(apk_node)(s)
def _CI(s):    return _wrap(ci_node)(s)
def _GE(s):    return _wrap(ge_node)(s)
# def _MH(s):    return _wrap(mh_node)(s)
def _AR(s):    return _wrap(ar_node)(s)
def _TC(s):    return _wrap(tc_node)(s)
def _RLC(s):   return _wrap(rlc_node)(s)
def _END(s):   return _wrap(end_node)(s)
def _AUTOSUGGESTION_MANAGER(s): return _wrap(autosuggestion_manager_node)(s)

# Pause node - just passes through to allow interrupt
def pause_for_handler(state: AgentState) -> AgentState:
    """Simple pass-through node that allows graph to interrupt after handler.
    
    Clears autosuggestions so user sees only the handler output without
    stale suggestions. Fresh autosuggestions will be generated when flow resumes.
    """
    state["handler_triggered"] = False  # Reset flag
    state["autosuggestions"] = []  # Clear final autosuggestions
    state["positive_autosuggestion"] = ""  # Clear positive selection
    state["negative_autosuggestion"] = ""  # Clear negative selection
    state["special_handling_autosuggestion"] = ""  # Clear special handling selection
    state["dynamic_autosuggestion"] = ""  # Clear dynamic suggestion
    return state

def _PAUSE(s): return _wrap(pause_for_handler)(s)

# NEW: Node wrappers (simulation)
def _SIM_CC(s):       return _wrap(sim_concept_creator_node)(s)
# def _SIM_VARS(s):     return _wrap(sim_vars_node)(s)
# def _SIM_ACTION(s):   return _wrap(sim_action_node)(s)
# def _SIM_EXPECT(s):   return _wrap(sim_expect_node)(s)
# def _SIM_EXECUTE(s):  return _wrap(sim_execute_node)(s)
# def _SIM_OBSERVE(s):  return _wrap(sim_observe_node)(s)
# def _SIM_INSIGHT(s):  return _wrap(sim_insight_node)(s)
# # def _SIM_NEXT(s):     return _wrap(sim_next_concept_node)(s)
# def _SIM_REFLECT(s):  return _wrap(sim_reflection_node)(s)

# -----------------------------------------------------------------------------
# // 5. Build the StateGraph
# -----------------------------------------------------------------------------
g = StateGraph(AgentState)
# g.add_node("INIT", _INIT)
g.add_node("START", _START)
g.add_node("APK", _APK)
g.add_node("CI",  _CI)
g.add_node("GE",  _GE)
# g.add_node("MH",  _MH)
g.add_node("AR",  _AR)
g.add_node("TC",  _TC)
g.add_node("RLC", _RLC)
g.add_node("END", _END)
g.add_node("AUTOSUGGESTION_MANAGER", _AUTOSUGGESTION_MANAGER)
g.add_node("PAUSE_FOR_HANDLER", _PAUSE)

# ▶ NEW: Simulation nodes
g.add_node("SIM_CC", _SIM_CC)
# g.add_node("SIM_VARS", _SIM_VARS)
# g.add_node("SIM_ACTION", _SIM_ACTION)
# g.add_node("SIM_EXPECT", _SIM_EXPECT)
# g.add_node("SIM_EXECUTE", _SIM_EXECUTE)
# g.add_node("SIM_OBSERVE", _SIM_OBSERVE)
# g.add_node("SIM_INSIGHT", _SIM_INSIGHT)
# # g.add_node("SIM_NEXT", _SIM_NEXT)
# g.add_node("SIM_REFLECT", _SIM_REFLECT)

def _route(state: AgentState) -> str:
    return state.get("current_state")

def _route_with_manager_check(state: AgentState) -> str:
    """Route to manager if autosuggestion was clicked, otherwise continue normal flow."""
    current_state = state.get("current_state")
    clicked = state.get("clicked_autosuggestion", False)
    
    if clicked:
        # User clicked autosuggestion - route through manager
        print(f"➡️ Routing to AUTOSUGGESTION_MANAGER from {current_state} due to clicked autosuggestion")
        return f"{current_state}_TO_MANAGER"
    else:
        # User typed - continue normal pedagogical flow
        print(f"➡️ Continuing to {current_state} without AUTOSUGGESTION_MANAGER")
        return current_state

def _route_after_manager(state: AgentState) -> str:
    """
    Route from AUTOSUGGESTION_MANAGER back to the appropriate pedagogical node.
    If handler was triggered, go to PAUSE_FOR_HANDLER (which interrupts).
    Otherwise, continue directly to the pedagogical node.
    """
    current_state = state.get("current_state")
    handler_triggered = state.get("handler_triggered", False)
    
    if handler_triggered:
        # Handler modified output - pause to show user
        print(f"⏸️ Routing to PAUSE_FOR_HANDLER after AUTOSUGGESTION_MANAGER")
        return f"{current_state}_PAUSED"
    else:
        # Normal autosuggestion - continue flow
        print(f"➡️ Continuing to {current_state} after AUTOSUGGESTION_MANAGER")
        return current_state

# g.add_edge(START, "INIT")
# g.add_edge("INIT", "START")
g.add_edge(START,"START")

g.add_edge("START","APK")

# Core flow - pedagogical nodes conditionally route to manager based on clicked_autosuggestion
g.add_conditional_edges(
    "APK", 
    _route_with_manager_check, 
    {"APK": "APK", "CI": "CI", "APK_TO_MANAGER": "AUTOSUGGESTION_MANAGER", "CI_TO_MANAGER": "AUTOSUGGESTION_MANAGER"}
)
g.add_conditional_edges(
    "CI", 
    _route_with_manager_check, 
    {"CI": "CI", "SIM_CC": "SIM_CC", "CI_TO_MANAGER": "AUTOSUGGESTION_MANAGER", "SIM_CC_TO_MANAGER": "AUTOSUGGESTION_MANAGER"}
)
g.add_conditional_edges(
    "GE", 
    _route_with_manager_check, 
    {"GE": "GE", "AR": "AR", "GE_TO_MANAGER": "AUTOSUGGESTION_MANAGER", "AR_TO_MANAGER": "AUTOSUGGESTION_MANAGER"}
)
g.add_conditional_edges(
    "AR", 
    _route_with_manager_check, 
    {"AR": "AR", "TC": "TC", "GE": "GE", "AR_TO_MANAGER": "AUTOSUGGESTION_MANAGER", "TC_TO_MANAGER": "AUTOSUGGESTION_MANAGER", "GE_TO_MANAGER": "AUTOSUGGESTION_MANAGER"}
)
g.add_conditional_edges(
    "TC", 
    _route_with_manager_check, 
    {"TC": "TC", "RLC": "RLC", "TC_TO_MANAGER": "AUTOSUGGESTION_MANAGER", "RLC_TO_MANAGER": "AUTOSUGGESTION_MANAGER"}
)
g.add_conditional_edges(
    "RLC", 
    _route_with_manager_check, 
    {"RLC": "RLC", "END": "END", "RLC_TO_MANAGER": "AUTOSUGGESTION_MANAGER", "END_TO_MANAGER": "AUTOSUGGESTION_MANAGER"}
)
g.add_edge("END", END)

# Manager routes back to pedagogical nodes OR pause node based on handler_triggered
g.add_conditional_edges(
    "AUTOSUGGESTION_MANAGER", 
    _route_after_manager, 
    {
        "APK": "APK", "CI": "CI", "GE": "GE", "AR": "AR", "TC": "TC", "RLC": "RLC","SIM_CC": "SIM_CC",
        "APK_PAUSED": "PAUSE_FOR_HANDLER", "CI_PAUSED": "PAUSE_FOR_HANDLER", 
        "GE_PAUSED": "PAUSE_FOR_HANDLER", "AR_PAUSED": "PAUSE_FOR_HANDLER", 
        "TC_PAUSED": "PAUSE_FOR_HANDLER", "RLC_PAUSED": "PAUSE_FOR_HANDLER"
    }
)

# Pause node routes back to the actual pedagogical node
g.add_conditional_edges("PAUSE_FOR_HANDLER", _route, {"APK": "APK", "CI": "CI","SIM_CC": "SIM_CC", "GE": "GE", "AR": "AR", "TC": "TC", "RLC": "RLC"})

# Simulation flow edges
g.add_conditional_edges("SIM_CC", _route, {"GE": "GE"})
# g.add_edge("SIM_CC", "SIM_VARS")
# g.add_edge("SIM_VARS", "SIM_ACTION")
# g.add_edge("SIM_ACTION", "SIM_EXPECT")
# g.add_edge("SIM_EXPECT", "SIM_EXECUTE")
# g.add_edge("SIM_EXECUTE", "SIM_OBSERVE")
# g.add_edge("SIM_OBSERVE", "SIM_INSIGHT")
# g.add_edge("SIM_INSIGHT", "SIM_REFLECT")
# g.add_edge("SIM_REFLECT", "AR")   # After simulation, go to AR to ask question about the concept
# g.add_edge("SIM_VARS", "AR")

# checkpointer = InMemorySaver()
checkpointer = SqliteSaver.from_conn_string("sqlite:///./.lg_memory.db")

# Initialize PostgreSQL checkpointer
try:
    connection_kwargs = {
        "autocommit": True,  # Required for Transaction Mode
        "prepare_threshold": None,  # None = Never use prepared statements (required for Transaction Mode port 6543)
        "gssencmode": "disable",  # Critical for Windows: prevents GSSAPI negotiation that Supabase doesn't support
    }
    
    postgres_url = os.getenv('POSTGRES_DATABASE_URL')
    print(f"🔍 Initializing Postgres checkpointer with URL: {postgres_url}")
    if not postgres_url:
        raise ValueError("POSTGRES_DATABASE_URL environment variable is not set")
    
    # IMPORTANT: Assume tables are already created (skip setup for Transaction Mode compatibility)
    # Tables must be created beforehand via Supabase dashboard or setup_postgres_tables.py
    skip_setup = os.getenv('SKIP_POSTGRES_SETUP', 'true').lower() == 'true'  # Default to TRUE
    
    pool = ConnectionPool(
        conninfo=postgres_url,
        max_size=40,  # Stay within Supabase Transaction Mode limits (set to 42 on dashboard)
        min_size=5,   # Reduced for Transaction Mode efficiency
        timeout=30,   # Wait up to 30s for available connection
        # === CONNECTION LIFECYCLE MANAGEMENT (fixes SSL/DbHandler errors) ===
        max_idle=300,        # Close connections idle > 5 min (before Supabase closes them)
        max_lifetime=1800,   # Recycle ALL connections every 30 min (fresh SSL sessions)
        reconnect_timeout=30,  # Retry failed connections for up to 30s
        kwargs=connection_kwargs,
    )
    checkpointer = PostgresSaver(pool)
    
    if not skip_setup:
        print("🔧 Running checkpointer.setup() to create tables...")
        checkpointer.setup()  # Create tables if they don't exist
        print("✅ Tables created/verified")
    else:
        print("⏭️  Skipping table setup (assuming tables exist)")
    
    print("✅ Postgres checkpointer initialized successfully (with connection lifecycle management)")
except Exception as e:
    print(f"❌ Error initializing Postgres checkpointer: {e}")
    print(f"💡 Ensure tables exist: checkpoints, checkpoint_writes, checkpoint_migrations")
    raise e

def build_graph():
    compiled = g.compile(
        checkpointer=checkpointer,
        interrupt_after=[
            "START", "PAUSE_FOR_HANDLER",  # Interrupt after START and after handler output
            # ▶ NEW: pause points for simulation path
            "APK", "CI", "GE", "AR", "TC", "RLC",
            "SIM_CC",
        ],
    )
    return compiled

graph = build_graph()
