# Math Tutoring Agent with Adaptive Pedagogy

An intelligent math tutoring system built with LangGraph that adapts its teaching approach in real-time based on student understanding (Tu) and approach quality (Ta).

## 🎯 Overview

This agent implements a sophisticated pedagogical framework that routes students through different teaching modes based on continuous assessment:

- **COACH Mode**: For students with strong understanding (Ta≥0.6, Tu≥0.6) - uses reflective questions
- **GUIDED Mode**: For students with partial understanding - provides targeted hints
- **SCAFFOLD Mode**: For students needing step-by-step guidance (Ta<0.6, Tu<0.6) - breaks down into concrete operations
- **CONCEPT Mode**: Teaches missing prerequisite concepts on-the-fly

## 📁 Project Structure

```
educational_agent_math_tutor/
├── __init__.py           # Package initialization
├── graph.py              # LangGraph workflow definition
├── nodes.py              # Node implementations (START, ASSESSMENT, ADAPTIVE_SOLVER, REFLECTION)
├── schemas.py            # State and Pydantic response models
├── prompts.py            # Comprehensive teaching prompts
├── config.py             # Configuration (thresholds, limits)
└── test_agent.py         # Test script

utils/
├── __init__.py
└── shared_utils.py       # LLM invocation, JSON extraction, problem loading

problems_json/
└── Fractions.json        # Problem definitions
```

## 🔄 Agent Flow

```
START
  ↓
  Load problem from JSON
  Present to student & ask understanding/approach
  ↓
ASSESSMENT
  ↓
  Evaluate Tu (Understanding: 0-1) using rubric:
    - Identifies operation needed?
    - Understands problem terms?
    - Knows what result represents?
  
  Evaluate Ta (Approach: 0-1) using rubric:
    - Correct method mentioned?
    - Logical step order?
    - Handles edge cases?
  
  Detect missing prerequisite concepts
  ↓
  Route to mode:
    - missing_concept → CONCEPT
    - Ta≥0.6 & Tu≥0.6 → COACH
    - Ta<0.6 & Tu<0.6 → SCAFFOLD
    - else → GUIDED
  ↓
ADAPTIVE_SOLVER
  ↓
  [COACH]
    - Validate work, praise effort
    - If wrong: Ask "why" questions (max 3 nudges)
    - Still wrong after 3? Downgrade to GUIDED
  
  [GUIDED]
    - Acknowledge what's right
    - Point out missing piece
    - Provide clear hint
  
  [SCAFFOLD]
    - Give ONE concrete operation per step
    - Check understanding after each
    - Failed after MAX_RETRIES? Give answer & move to next step
  
  [CONCEPT]
    - Teach missing concept with age-appropriate analogy
    - Micro-check question
    - Resume previous mode
  ↓
  Solved? → REFLECTION
  Not solved? → Loop back to ADAPTIVE_SOLVER
  ↓
REFLECTION
  ↓
  Celebrate success
  Confidence check
  Suggest next steps
  ↓
END
```

## 🚀 Usage

### Basic Usage

```python
from langchain_core.messages import HumanMessage
from educational_agent_math_tutor import graph

# Configure session
config = {
    "configurable": {
        "thread_id": "student_session_1"
    }
}

# Initialize with problem
initial_state = {
    "problem_id": "add_frac_same_den_01",
    "messages": [],
}

result = graph.invoke(initial_state, config)

# Student responds
result = graph.invoke(
    {
        "messages": [HumanMessage(content="I think I add the numerators...")]
    },
    config
)

# Continue conversation...
```

### Running the Test

```bash
cd educational_agent_math_tutor
python test_agent.py
```

## 📊 State Fields

### Core Fields

- `problem`: Problem text
- `problem_id`: Identifier for loading from JSON
- `Ta`: Approach quality score (0-1)
- `Tu`: Understanding quality score (0-1)
- `mode`: Current pedagogical mode
- `solved`: Whether problem is complete

### Tracking Fields

- `steps`: Solution steps from JSON (with concept info)
- `step_index`: Current step in scaffold mode
- `nudge_count`: Reflective questions asked in coach mode
- `scaffold_retry_count`: Failed attempts on current step
- `node_transitions`: List of transitions with timestamps

### Future Enhancement Fields

- `summary`: Rolling conversation summary (unused for now)
- `summary_last_index`: Last summarized message index (unused for now)

## 🎨 Pedagogical Design

### Assessment Rubrics

**Tu (Understanding) - 3 criteria:**

1. Identifies what operation is needed
2. Understands problem terms/meaning
3. Knows what result represents

**Ta (Approach) - 3 criteria:**

1. Mentions correct method
2. Logical step order
3. Handles conversions/edge cases

### Mode Thresholds (configurable in config.py)

- `TA_THRESHOLD_HIGH = 0.6`
- `TU_THRESHOLD_HIGH = 0.6`
- `MAX_COACH_NUDGES = 3`
- `MAX_SCAFFOLD_RETRIES = 2`

## 📝 Problem JSON Format

```json
{
  "problem_id": "add_frac_same_den_01",
  "topic": "Addition of Fractions (Same Denominator)",
  "difficulty": "easy",
  "question": "Add: 3/8 + 2/8",
  "final_answer": "5/8",
  "canonical_solution": {
    "steps": [
      {
        "step_id": 1,
        "description": "Check the denominators of both fractions.",
        "concept": "fraction_basics"
      },
      {
        "step_id": 2,
        "description": "Since the denominators are the same, add the numerators.",
        "concept": "addition_same_denominator"
      },
      {
        "step_id": 3,
        "description": "Add numerators: 3 + 2 = 5, keep denominator 8.",
        "concept": "addition_same_denominator"
      }
    ]
  },
  "required_concepts": ["fraction_basics", "addition_same_denominator"]
}
```

## 🔧 Configuration

Edit [config.py](educational_agent_math_tutor/config.py) to adjust:

- Ta/Tu thresholds for mode routing
- Maximum coach nudges before downgrade
- Maximum scaffold retries per step
- Default LLM model and temperature

## 🛠️ Future Enhancements

### Memory Optimization (Prepared but Not Implemented)

The wrapper pattern in [graph.py](educational_agent_math_tutor/graph.py) is designed to easily add:

- Conversation summarization
- Node-aware history segmentation
- Rolling summary updates

To enable later, enhance the wrapper to:

1. Capture old state before node call
2. Detect new user messages
3. Build summaries using `node_transitions`
4. Populate `summary` and `summary_last_index` fields

### Other Enhancements

- Autosuggestions (positive/negative/special handling)
- Multi-language support
- Voice/OCR input normalization
- Session metrics tracking
- A/B testing of different thresholds

## 📚 Dependencies

- `langgraph>=0.2.30`
- `langchain>=0.2.7`
- `langchain-google-genai>=2.0.0`
- `pydantic` (for structured outputs)

Optional:

- `api_tracker_utils` (for API key management and rate limiting)

## 🧪 Testing Different Scenarios

The agent handles various student scenarios:

1. **Strong Student**: High Ta & Tu → COACH mode → reflective questions
2. **Confused Student**: Low Ta & Tu → SCAFFOLD mode → step-by-step
3. **Partial Understanding**: Mixed Ta/Tu → GUIDED mode → targeted hints
4. **Missing Concept**: Detected gap → CONCEPT mode → teach prerequisite
5. **Persistent Struggle**: Coach nudges exhausted → downgrade to GUIDED

## 📖 Example Interaction

```
AGENT: What do you understand from this question? What approach would you use?

STUDENT: I need to add fractions. I'll add the numerators and denominators.

AGENT: [ASSESSMENT] Tu=0.5, Ta=0.3 → GUIDED MODE
       I see you understand we're adding fractions! However, adding denominators 
       isn't quite right. Think about what the denominator represents...

STUDENT: Oh! The denominator shows how many parts. So I keep it the same?

AGENT: [COACH MODE] Exactly! Now try solving it.

STUDENT: 3 + 2 = 5, keep 8. Answer is 5/8!

AGENT: [REFLECTION] Excellent work! You figured out that when denominators 
       are the same, we only add the numerators...
```

## 🤝 Contributing

To add new problems:

1. Add JSON file to `problems_json/`
2. Follow the canonical structure
3. Include steps with concepts
4. Test with `test_agent.py`
