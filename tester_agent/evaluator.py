import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from tester_agent.personas import Persona
from tester_agent.session_metrics import compute_and_upload_session_metrics
from typing import Optional, Dict, Any
from langchain_groq import ChatGroq

class Evaluator:
    """
    Educational Quality Evaluator - Focuses on pedagogical effectiveness and qualitative assessment.
    
    This evaluator complements the quantitative session metrics by providing:
    - Educational quality assessment (pedagogical flow, scaffolding, etc.)
    - Qualitative feedback for educational improvement
    - Technical issue identification
    
    Note: For quantitative metrics (engagement, clarity scores, etc.), use session_metrics.py
    """
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-3-27b-it",
            api_key=os.getenv("GOOGLE_API_KEY_4"),
            temperature=0.2,
        )
    #     self.llm = ChatGroq(
    #     model="llama-3.1-8b-instant",
    #     temperature=0.5,
    #     max_tokens=None,
    # )

    def evaluate(self, persona: Persona, history: list) -> str:
        """
        Evaluates the educational quality and pedagogical effectiveness of the conversation.
        Focuses on qualitative assessment complementary to quantitative session metrics.
        """
        prompt = f"""
You are an expert in evaluating educational conversations for pedagogical effectiveness.
Your task is to analyze the following conversation between an educational agent and a student with the persona of a "{persona.name}".

**Persona Description:** {persona.description}

**Conversation History:**
{json.dumps(history, indent=2)}
    
**Educational Quality Metrics:**
- **Pedagogical Flow:** Did the conversation follow a logical and effective teaching structure? Did it progress from simple to complex concepts appropriately? (1-5)
- **Learning Objective Achievement:** How well did the agent help the student achieve the learning objectives? Were concepts properly introduced and reinforced? (1-5)
- **Scaffolding Effectiveness:** Did the agent provide appropriate support and gradually reduce guidance as the student progressed? (1-5)
- **Misconception Handling:** How effectively did the agent identify and address student misconceptions? (1-5)

**Qualitative Feedback:**
- **Pedagogical Strengths:** What educational strategies did the agent use effectively?
- **Areas for Educational Improvement:** What pedagogical approaches could be enhanced?
- **Technical Issues:** Were there any bugs, technical problems, or system errors?
- **Persona Alignment:** How well did the agent adapt its teaching style to this specific persona?

**Output Format:**
Please provide your evaluation in JSON format. Give me the json object directly WITHOUT any additional text like ```json in the beginning etc.
I want to run the function json.loads on your output directly.

"""
        response = self.llm.invoke(prompt)
        return response.content
    
    def evaluate_with_metrics(self, 
                            persona: Persona, 
                            history: list, 
                            session_id: str,
                            session_state: Dict[str, Any],
                            upload_metrics: bool = True) -> Dict[str, Any]:
        """
        Provides comprehensive assessment combining educational quality evaluation 
        with quantitative session metrics.
        
        Args:
            persona: Student persona used in the conversation
            history: Conversation history
            session_id: Unique session identifier
            session_state: Final agent state
            upload_metrics: Whether to compute and upload quantitative metrics to Langfuse
            
        Returns:
            Dictionary containing:
            - Educational quality evaluation (pedagogical effectiveness)
            - Quantitative session metrics (engagement, performance, etc.)
            - Persona and history data
        """
        # Get regular evaluation
        evaluation_str = self.evaluate(persona, history)
        
        # Parse evaluation
        clean_str = evaluation_str.strip()
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]
        clean_str = clean_str.strip()
        
        result = {
            "persona": persona.model_dump(),
            "educational_evaluation": json.loads(clean_str),  # Pedagogical quality assessment
            "history": history,
        }
        
        # Optionally compute and upload quantitative session metrics
        if upload_metrics:
            try:
                session_metrics = compute_and_upload_session_metrics(
                    session_id=session_id,
                    history=history,
                    session_state=session_state,
                    persona_name=persona.name
                )
                result["session_metrics"] = session_metrics.model_dump()  # Quantitative analytics
                print(f"✅ Educational evaluation and session metrics completed for: {session_id}")
            except Exception as e:
                print(f"❌ Failed to compute/upload session metrics: {e}")
                result["metrics_error"] = str(e)
        
        return result