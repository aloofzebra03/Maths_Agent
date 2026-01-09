# Installation & Setup Guide

## 📦 Installation

### 1. Install Dependencies

```bash
pip install langgraph langchain langchain-google-genai python-dotenv pydantic
```

Or install all dependencies from the project:

```bash
pip install -e .
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Verify Installation

```bash
cd educational_agent_math_tutor
python test_agent.py
```

## 🚀 Quick Start

### Run the Test Session

```bash
python educational_agent_math_tutor/test_agent.py
```

This will run a sample tutoring session with the problem "Add: 3/8 + 2/8".

### Use in Your Own Code

```python
from langchain_core.messages import HumanMessage
from educational_agent_math_tutor import graph

# Configure session
config = {"configurable": {"thread_id": "session_1"}}

# Initialize with problem
result = graph.invoke(
    {"problem_id": "add_frac_same_den_01", "messages": []},
    config
)

# Continue conversation
result = graph.invoke(
    {"messages": [HumanMessage(content="Your response...")]},
    config
)
```

## 🔍 Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure:
1. You're in the project root directory
2. Dependencies are installed: `pip install -e .`
3. Python path includes the project root

### API Key Issues

If you get authentication errors:
1. Check `.env` file exists in project root
2. Verify `GOOGLE_API_KEY` is set correctly
3. Ensure `python-dotenv` is installed

### JSON Loading Errors

If problems can't be loaded:
1. Verify `Fractions.json` is in `problems_json/` folder
2. Check JSON formatting is valid
3. Ensure problem_id matches the ID in JSON file

## 📝 Adding New Problems

1. Create or edit a JSON file in `problems_json/`
2. Use the format from `Fractions.json` as template
3. Include all required fields:
   - `problem_id`
   - `question`
   - `final_answer`
   - `canonical_solution` with `steps`
   - `required_concepts`

Example:

```json
{
  "problem_id": "your_problem_id",
  "topic": "Your Topic",
  "difficulty": "easy",
  "question": "Your question here",
  "final_answer": "Answer",
  "canonical_solution": {
    "steps": [
      {
        "step_id": 1,
        "description": "First step...",
        "concept": "concept_name"
      }
    ]
  },
  "required_concepts": ["concept1", "concept2"]
}
```

## 🧪 Development

### Running Tests

```bash
python educational_agent_math_tutor/test_agent.py
```

### Modifying Prompts

Edit `educational_agent_math_tutor/prompts.py` to customize:
- Teaching style
- Language complexity
- Pedagogical strategies

### Adjusting Thresholds

Edit `educational_agent_math_tutor/config.py` to tune:
- Ta/Tu thresholds for mode routing
- Maximum nudges/retries
- LLM temperature

### Debugging

Enable detailed logging in nodes by checking console output. Each node prints:
- Current state
- Assessment scores
- Mode routing decisions
- LLM responses

## 📚 Next Steps

1. Test with different problems
2. Adjust prompts based on student interactions
3. Tune thresholds for optimal pedagogy
4. Add more problems to `problems_json/`
5. Consider adding memory optimization (see README)

## 🆘 Getting Help

- Check the [README.md](README.md) for architecture details
- Review node implementations in `nodes.py`
- Examine prompts in `prompts.py`
- Test with simpler problems first
