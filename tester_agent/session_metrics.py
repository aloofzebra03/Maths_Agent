import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langfuse import get_client


# Pydantic model for LLM-analyzed metrics
class LLMAnalyzedMetrics(BaseModel):

    concepts_covered: List[str] = Field(description="List of educational concepts taught during the session")
    clarity_conciseness_score: float = Field(description="Score 1-5 for how clear and concise the agent's explanations were", ge=1, le=5)
    user_type: Literal["Dull", "Medium", "High"] = Field(description="Learner classification based on comprehension speed and curiosity")
    user_interest_rating: float = Field(description="Score 1-5 based on enthusiasm, questions asked, and voluntary engagement", ge=1, le=5)
    user_engagement_rating: float = Field(description="Score 1-5 based on response quality, participation, and willingness to explore", ge=1, le=5)
    enjoyment_probability: float = Field(description="Probability 0-1 that the user enjoyed and benefited from the session", ge=0, le=1)
    error_handling_count: int = Field(description="Count of times the agent had to clarify, correct, or re-explain something", ge=0)
    adaptability: bool = Field(description="Whether the agent adapted its teaching approach based on user responses")


class SessionMetrics(BaseModel):
    """Complete session-level metrics for educational agent interactions"""
    
    # LLM-analyzed metrics
    concepts_covered: List[str] = Field(description="List of concepts taught during the session")
    num_concepts_covered: int = Field(description="Total number of concepts touched")
    clarity_conciseness_score: float = Field(description="Based on LLM evaluation (1-5)", ge=1, le=5)
    user_type: str = Field(description="Categorized as Dull, Medium, or High learner")
    user_interest_rating: float = Field(description="Score (1-5) based on engagement indicators", ge=1, le=5)
    user_engagement_rating: float = Field(description="Score (1-5) using response patterns", ge=1, le=5)
    enjoyment_probability: float = Field(description="Likelihood (0-1) that user enjoyed and benefited", ge=0, le=1)
    
    # Computed metrics
    quiz_score: float = Field(description="Score from formative assessments (0-100)", ge=0, le=100)
    error_handling_count: int = Field(description="Count of corrections/re-prompts", ge=0)
    adaptability: bool = Field(description="Whether flow was adjusted dynamically to user performance")
    
    # Session metadata
    session_id: str = Field(description="Unique session identifier")
    total_interactions: int = Field(description="Total number of user-agent exchanges", ge=0)
    persona_name: Optional[str] = Field(description="User persona if known")

# --- Score Configs (enforce schema; replace with your real config IDs) ---
SCORE_CONFIGS = {
    "num_concepts_covered": "cmeuwebal024tad082cqip658",
    "clarity_conciseness_score": "cmeuwfcmx024wad08mrqcs430",
    "user_type": "cmeuwh5r300bsad0752kaermc",
    "user_interest_rating": "cmeuwicno0001ad06awfm3nro",
    "user_engagement_rating": "cmeuwj89z003aad07q7c1xiqe",
    "enjoyment_probability": "cmeuwjt270005ad0644xa4tvo",
    "quiz_score": "cmeuwkn070008ad066edbbqjj",
    "error_handling_count": "cmeuwl304000bad06tj4938z0",
    "adaptability": "cmeuwluud003ead079uzva6oi",
    "total_interactions": "cmeuwm85f00niad07mzbi97sn",
}


class MetricsComputer:
    """Computes session metrics from conversation history and state"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-3-27b-it",
            api_key=os.getenv("GOOGLE_API_KEY_5"),
            temperature=0.2,
        )
        self.langfuse = get_client()
        self.llm_parser = PydanticOutputParser(pydantic_object=LLMAnalyzedMetrics)
    
    def compute_metrics(self, 
                       session_id: str,
                       history: List[Dict[str, Any]], 
                       session_state: Dict[str, Any],
                       persona_name: Optional[str] = None) -> SessionMetrics:
        """
        Compute all session metrics from conversation history and agent state
        """
        
        # Extract basic session info
        user_interactions = [h for h in history if h.get("role") == "user"]
        agent_interactions = [h for h in history if h.get("role") == "assistant"]
        total_interactions = len(user_interactions)
    
        
        # Get LLM-analyzed metrics in one call
        llm_metrics = self._analyze_conversation_with_llm(history, persona_name)
        
        # Compute simple metrics that don't need LLM
        quiz_score = self._extract_quiz_score(history, session_state)
        
        return SessionMetrics(
            # LLM-analyzed metrics
            concepts_covered=llm_metrics.concepts_covered,
            num_concepts_covered=len(llm_metrics.concepts_covered),
            clarity_conciseness_score=llm_metrics.clarity_conciseness_score,
            user_type=llm_metrics.user_type,
            user_interest_rating=llm_metrics.user_interest_rating,
            user_engagement_rating=llm_metrics.user_engagement_rating,
            enjoyment_probability=llm_metrics.enjoyment_probability,
            error_handling_count=llm_metrics.error_handling_count,
            adaptability=llm_metrics.adaptability,
            
            # Computed metrics
            quiz_score=quiz_score,
            
            # Session metadata
            session_id=session_id,
            total_interactions=total_interactions,
            persona_name=persona_name
        )
    
    def _analyze_conversation_with_llm(self, history: List[Dict[str, Any]], persona_name: Optional[str] = None) -> LLMAnalyzedMetrics:
        
        # Format conversation for LLM analysis
        conversation_text = self._format_conversation_for_analysis(history)
        
        persona_context = f"\nThe user was acting with the persona: {persona_name}\n" if persona_name else ""
        
        prompt = f"""
You are an expert educational analyst. Analyze this educational conversation and provide structured metrics.

{persona_context}
**Conversation History:**
{conversation_text}

**Analysis Instructions:**
1. **Concepts Covered**: List the main educational concepts that were taught (e.g., "simple pendulum", "oscillation", "period", "frequency")
2. **Clarity & Conciseness**: Rate 1-5 how clear and concise the agent's explanations were
3. **User Type**: Classify as "Dull", "Medium", or "High" based on:
   - Speed of understanding
   - Quality of questions asked
   - Curiosity level
   - Response complexity
4. **Interest Rating**: Rate 1-5 the user's interest level based on:
   - Enthusiasm in responses
   - Questions asked voluntarily
   - Engagement indicators
5. **Engagement Rating**: Rate 1-5 the user's engagement based on:
   - Response length and quality
   - Willingness to participate
   - Active involvement
6. **Enjoyment Probability**: Estimate 0-1 likelihood that user enjoyed and benefited from the session
7. **Error Handling Count**: Count how many times the agent had to:
   - Clarify or re-explain something
   - Correct a misunderstanding
   - Provide additional help when user struggled
   - Rephrase or simplify explanations
8. **Adaptability**: Determine if the agent adapted its teaching approach based on user responses:
   - Did it adjust difficulty level?
   - Did it provide more examples when user struggled?
   - Did it change explanations based on user feedback?
   - Did it modify the teaching flow based on user performance?

{self.llm_parser.get_format_instructions()}
"""
        
        try:
            response = self.llm.invoke(prompt)
            return self.llm_parser.parse(response.content)
        except Exception as e:
            print(f" Error: LLM analysis failed: {e}")
            raise RuntimeError(f"Failed to analyze conversation with LLM: {e}") from e
    
    def _format_conversation_for_analysis(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for LLM analysis"""
        formatted_lines = []
        for interaction in history:
            role = interaction.get("role", "unknown")
            content = interaction.get("content", "")
            
            # Handle string content in history
            if isinstance(content, str):
                formatted_lines.append(f"{role.title()}: {content}")
            else:
                formatted_lines.append(f"{role.title()}: [Non-text content]")
        
        return "\n".join(formatted_lines)
    
    def _extract_quiz_score(self, history: List[Dict], state: Dict) -> float:
        quiz_score = state.get("quiz_score")
        if quiz_score is not None:
            return float(quiz_score)
        
        return 0.0
    
    def upload_to_langfuse(self, metrics: SessionMetrics) -> bool:
        """
        Post-hoc: attach all metrics as scores to the Langfuse SESSION (no new trace).
        Enforces Score Configs by passing config_id per score.
        """
        try:
            md = metrics.model_dump()

            def _put(name: str, value, data_type: str, comment: Optional[str] = None):
                payload = {
                    "name": name,
                    "session_id": metrics.session_id,      # <-- SESSION target
                    "config_id": SCORE_CONFIGS.get(name), # <-- enforce config
                    "data_type": data_type,               # explicit is fine
                }
                if data_type == "CATEGORICAL":
                    payload["value"] = str(value)
                elif data_type in ("NUMERIC", "BOOLEAN"):
                    payload["value"] = float(value)
                if comment:
                    payload["comment"] = comment
                self.langfuse.create_score(**payload)

            # concepts -> count (validated by "num_concepts_covered" config)
            _put(
                "num_concepts_covered",
                len(md.get("concepts_covered", []) or []),
                "NUMERIC",
                comment=str(md.get("concepts_covered", [])),
            )

            # numeric scores
            for name in [
                "clarity_conciseness_score",
                "user_interest_rating",
                "user_engagement_rating",
                "enjoyment_probability",
                "quiz_score",
                "error_handling_count",
                "total_interactions",
            ]:
                val = md.get(name)
                if isinstance(val, (int, float)):
                    _put(name, float(val), "NUMERIC")

            # boolean
            _put("adaptability", 1.0 if md.get("adaptability") else 0.0, "BOOLEAN")

            # categorical
            if md.get("user_type"):
                _put("user_type", str(md["user_type"]), "CATEGORICAL")

            # flush for short-lived scripts
            self.langfuse.flush()

            print(f"✅ Uploaded session-level scores (enforced by Score Configs) for session: {metrics.session_id}")
            return True

        except Exception as e:
            print(f"❌ Error: Failed to upload metrics to Langfuse: {e}")
            raise RuntimeError(f"Failed to upload metrics to Langfuse: {e}") from e

def compute_and_upload_session_metrics(session_id: str, 
                                    history: List[Dict[str, Any]], 
                                    session_state: Dict[str, Any],
                                    persona_name: Optional[str] = None) -> SessionMetrics:
    """
    Convenience function to compute and upload session metrics
    
    Args:
        session_id: Unique session identifier
        history: Conversation history
        session_state: Agent's final state
        persona_name: Optional persona name for the user
    
    Returns:
        SessionMetrics object with all computed metrics
        
    Raises:
        RuntimeError: If metrics computation or upload fails
    """
    try:
        computer = MetricsComputer()
        metrics = computer.compute_metrics(session_id, history, session_state, persona_name)
        
        # Upload to Langfuse
        # computer.upload_to_langfuse(metrics)
        
        print(f"📊 Session metrics computed and uploaded successfully!")
        print(f"   - Concepts covered: {metrics.num_concepts_covered}")
        print(f"   - User type: {metrics.user_type}")
        print(f"   - Quiz score: {metrics.quiz_score:.1f}%")
        print(f"   - Engagement rating: {metrics.user_engagement_rating:.1f}/5")
        print(f"   - Enjoyment probability: {metrics.enjoyment_probability:.2f}")
        
        return metrics
    
    except Exception as e:
        print(f"❌ Error: Session metrics computation failed: {e}")
        raise RuntimeError(f"Session metrics computation failed: {e}") from e
