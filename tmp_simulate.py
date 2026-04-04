"""
Simulation script — tests scaffold logic directly without the full graph.
Verifies:
  1. Correct answer → immediate step advance (no reveal)
  2. Wrong answers → retry, then reveal at MAX_SCAFFOLD_RETRIES
  3. ASSESS_APPROACH is NOT called between scaffold steps (graph routing check)
  4. Coach mode still routes to ASSESS_APPROACH
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage, AIMessage
from educational_agent_math_tutor.nodes import _scaffold_logic
from educational_agent_math_tutor.graph import should_continue_solving
from educational_agent_math_tutor.config import MAX_SCAFFOLD_RETRIES

STEPS = [
    {"step_id": 1, "description": "Identify the signs: 4 is positive, -6 is negative.", "concept": "identifying_signs"},
    {"step_id": 2, "description": "Apply sign rule: positive × negative = negative.", "concept": "integer_multiplication_rules"},
    {"step_id": 3, "description": "Multiply magnitudes: 4 × 6 = 24. Apply negative sign → -24.", "concept": "final_calculation"},
]

def base_state(step_index=0, retry_count=0, messages=None):
    return {
        "problem": "Evaluate: 4 × (-6)",
        "problem_id": "integer_mult_signs_01",
        "steps": STEPS,
        "step_index": step_index,
        "scaffold_retry_count": retry_count,
        "mode": "scaffold",
        "solved": False,
        "is_kannada": False,
        "messages": messages or [HumanMessage(content="I don't know, I'm confused")],
        "current_state": "ADAPTIVE_SOLVER",
    }

SEP = "═" * 60

# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SIMULATION 1: Happy Path — student answers Step 1 correctly")
print(SEP)

# Turn 1: retry_count=0 → should present first instruction
print("\n[Turn 1] retry_count=0 → expect: first instruction presented, retry_count→1")
state = base_state(step_index=0, retry_count=0)
result = _scaffold_logic(state)
print(f"  → scaffold_retry_count = {result.get('scaffold_retry_count')}")
print(f"  → step_index = {result.get('step_index', 0)}")
print(f"  → agent_output = {result.get('agent_output', '')[:120]}...")
assert result.get("scaffold_retry_count") == 1, "❌ FAIL: should set retry_count=1 after first instruction"
assert result.get("step_index", 0) == 0, "❌ FAIL: step_index should still be 0"
print("  → ✅ PASS: First instruction presented correctly")

# Turn 2: student answers correctly → should_advance=True, step advances
print(f"\n[Turn 2] retry_count=1, student gives correct answer → expect: step_index→1, retry_count→0")
state2 = base_state(
    step_index=0,
    retry_count=1,
    messages=[
        HumanMessage(content="I don't know"),
        AIMessage(content="Let's look at the signs — 4 is positive and -6 is negative. Can you tell me which one is positive?"),
        HumanMessage(content="4 is positive and -6 is negative"),
    ]
)
result2 = _scaffold_logic(state2)
print(f"  → scaffold_retry_count = {result2.get('scaffold_retry_count')}")
print(f"  → step_index = {result2.get('step_index', state2['step_index'])}")
print(f"  → agent_output = {result2.get('agent_output', '')[:120]}...")
step_advanced = result2.get("step_index", 0) == 1
retry_reset = result2.get("scaffold_retry_count") == 0
if step_advanced and retry_reset:
    print("  → ✅ PASS: Step advanced to 1, retry_count reset to 0")
else:
    print(f"  → ❌ FAIL: step_index={result2.get('step_index',0)}, retry_count={result2.get('scaffold_retry_count')}")

# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SIMULATION 2: Wrong answers → reveal at MAX_SCAFFOLD_RETRIES")
print(SEP)
print(f"MAX_SCAFFOLD_RETRIES = {MAX_SCAFFOLD_RETRIES}")

# Turn 1: First instruction
state_r0 = base_state(step_index=0, retry_count=0)
r0 = _scaffold_logic(state_r0)
print(f"\n[Turn 1] retry_count=0 → first instruction. retry_count→{r0.get('scaffold_retry_count')}")

# Turn 2: Wrong answer (retry_count=1)
msgs_wrong1 = [
    HumanMessage(content="I don't know"),
    AIMessage(content=r0.get("agent_output", "")),
    HumanMessage(content="Both numbers are positive"),
]
state_r1 = base_state(step_index=0, retry_count=1, messages=msgs_wrong1)
r1 = _scaffold_logic(state_r1)
print(f"[Turn 2] retry_count=1, WRONG → retry_count→{r1.get('scaffold_retry_count', 1)}, step_index={r1.get('step_index',0)}")
assert r1.get("step_index", 0) == 0, "❌ FAIL: should NOT advance step on wrong answer"

# Turn 3: Wrong again (retry_count=2=MAX) → should reveal
msgs_wrong2 = msgs_wrong1 + [
    AIMessage(content=r1.get("agent_output", "")),
    HumanMessage(content="I still think both are positive"),
]
state_r2 = base_state(step_index=0, retry_count=MAX_SCAFFOLD_RETRIES, messages=msgs_wrong2)
r2 = _scaffold_logic(state_r2)
print(f"[Turn 3] retry_count={MAX_SCAFFOLD_RETRIES}=MAX → REVEAL path")
print(f"  → step_index={r2.get('step_index', 0)} (should be 1)")
print(f"  → scaffold_retry_count={r2.get('scaffold_retry_count')} (should be 0)")
print(f"  → agent_output={r2.get('agent_output', '')[:150]}...")
if r2.get("step_index", 0) == 1 and r2.get("scaffold_retry_count") == 0:
    print("  → ✅ PASS: Reveal triggered, step advanced, retry reset")
else:
    print("  → ❌ FAIL")

# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SIMULATION 3: Graph routing — scaffold bypasses ASSESS_APPROACH")
print(SEP)

scaffold_state = {"solved": False, "mode": "scaffold"}
coach_state = {"solved": False, "mode": "coach"}
guided_state = {"solved": False, "mode": "guided"}
solved_state = {"solved": True, "mode": "scaffold"}

r_scaffold = should_continue_solving(scaffold_state)
r_coach = should_continue_solving(coach_state)
r_guided = should_continue_solving(guided_state)
r_solved = should_continue_solving(solved_state)

print(f"  scaffold (not solved) → '{r_scaffold}'  (expected: 'adaptive_solver')")
print(f"  coach    (not solved) → '{r_coach}'     (expected: 'assess_approach')")
print(f"  guided   (not solved) → '{r_guided}'    (expected: 'assess_approach')")
print(f"  scaffold (solved)     → '{r_solved}'    (expected: 'reflection')")

assert r_scaffold == "adaptive_solver", f"❌ FAIL: scaffold should bypass ASSESS_APPROACH"
assert r_coach == "assess_approach",    f"❌ FAIL: coach should use ASSESS_APPROACH"
assert r_guided == "assess_approach",   f"❌ FAIL: guided should use ASSESS_APPROACH"
assert r_solved == "reflection",        f"❌ FAIL: solved should route to reflection"
print("  → ✅ PASS: All routing correct")

# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SIMULATION 4: All steps complete → solved=True")
print(SEP)

all_done_state = base_state(step_index=len(STEPS), retry_count=0)
r_done = _scaffold_logic(all_done_state)
print(f"  solved={r_done.get('solved')} (expected: True)")
assert r_done.get("solved") == True, "❌ FAIL: should be solved when step_index >= len(steps)"
print("  → ✅ PASS: Completion message and solved=True")

print(f"\n{SEP}")
print("ALL SIMULATIONS COMPLETE")
print(SEP)
