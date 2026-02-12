# Dual LLM Implementation: Gemma vs Gemini

This project supports **two identical implementations** of the Math Tutoring Agent that differ **only in the LLM model used**:

1. **Base Implementation** (Gemma): Uses `gemma-3-27b-it`
2. **Gemini Implementation**: Uses `gemini-2.5-flash`

## Architecture Overview

Both implementations share identical:
- ✅ Graph structure and node logic
- ✅ Pedagogical strategies (coach/guided/scaffold modes)
- ✅ Assessment algorithms
- ✅ Concept teaching flow
- ✅ UI/UX interface

The **only difference** is the LLM model selection in the utilities layer.

## File Structure

The codebase follows a parallel file pattern:

### Core Agent Files

| Base (Gemma) | Gemini Variant | Difference |
|--------------|----------------|------------|
| [`educational_agent_math_tutor/graph.py`](educational_agent_math_tutor/graph.py) | [`educational_agent_math_tutor/graph_gemini.py`](educational_agent_math_tutor/graph_gemini.py) | Import source only |
| [`educational_agent_math_tutor/nodes.py`](educational_agent_math_tutor/nodes.py) | [`educational_agent_math_tutor/nodes_gemini.py`](educational_agent_math_tutor/nodes_gemini.py) | Import source only |
| [`utils/shared_utils.py`](utils/shared_utils.py) | [`utils/shared_utils_gemini.py`](utils/shared_utils_gemini.py) | **LLM model selection** |

### Streamlit UI Files

| Base (Gemma) | Gemini Variant | Difference |
|--------------|----------------|------------|
| [`streamlit_ui/app.py`](streamlit_ui/app.py) | [`streamlit_ui/app_gemini.py`](streamlit_ui/app_gemini.py) | Graph import + UI labels |

## Key Implementation Details

### 1. Graph Files
- **`graph.py`**: Imports from `nodes`
- **`graph_gemini.py`**: Imports from `nodes_gemini`
- All graph structure, routing logic, and interrupts are **100% identical**

### 2. Node Files
- **`nodes.py`**: Imports utilities from `shared_utils`
- **`nodes_gemini.py`**: Imports utilities from `shared_utils_gemini`
- All node logic, prompts, and pedagogical strategies are **100% identical**

### 3. Utility Files (LLM Selection Layer)

**`utils/shared_utils.py`**:
```python
def get_llm(api_key: Optional[str] = None, model: str = "gemma-3-27b-it") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )
```

**`utils/shared_utils_gemini.py`**:
```python
def get_llm(api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )
```

> **Note**: Both implementations use the same `ChatGoogleGenerativeAI` client from `langchain_google_genai`, just with different default models.

### 4. Streamlit UI Files

**`streamlit_ui/app.py`**:
- Imports from `educational_agent_math_tutor.graph`
- Page title: "Math Tutor Agent"
- UI labels: "Math Tutor", "Chat"

**`streamlit_ui/app_gemini.py`**:
- Imports from `educational_agent_math_tutor.graph_gemini`
- Page title: "Math Tutor Agent(Gemini)"
- UI labels: "Math Tutor(Gemini)", "Chat(Gemini)"
- Displays "**Model:** Gemini" in problem details

## Running the Application

### Start Base Implementation (Gemma)
```bash
streamlit run streamlit_ui/app.py
```

### Start Gemini Implementation
```bash
streamlit run streamlit_ui/app_gemini.py
```

Both UIs will:
- Load problems from `problems_json/`
- Support text and image input (with OCR)
- Provide identical tutoring experience
- Display debug information in the sidebar

## Why Two Implementations?

This parallel architecture allows for:

1. **A/B Model Testing**: Compare pedagogical effectiveness between Gemma and Gemini models
2. **Performance Benchmarking**: Measure response times, cost, and quality
3. **Model Fallback**: Easy switchover if one model experiences issues
4. **Research**: Study how different models handle the same tutoring strategies

## Adding New LLM Models

To add support for a new LLM (e.g., GPT-4, Claude):

1. Create new utility file: `utils/shared_utils_newmodel.py`
2. Update `get_llm()` function with new model configuration
3. Create parallel files:
   - `educational_agent_math_tutor/nodes_newmodel.py` (import from new utils)
   - `educational_agent_math_tutor/graph_newmodel.py` (import from new nodes)
   - `streamlit_ui/app_newmodel.py` (import from new graph)
4. No changes needed to core logic, prompts, or pedagogical strategies

---

**Summary**: The codebase maintains complete feature parity between implementations through a clean layered architecture where LLM selection is isolated to the utilities layer.
