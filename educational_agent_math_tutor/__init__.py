"""
Math Tutoring Agent with Adaptive Pedagogy.

This package implements a LangGraph-based math tutoring system that adapts
its teaching approach based on real-time assessment of student understanding (Tu)
and approach quality (Ta).

Pedagogical Modes:
    - COACH: For students with strong understanding (Ta≥0.6, Tu≥0.6)
    - GUIDED: For students with partial understanding
    - SCAFFOLD: For students needing step-by-step guidance (Ta<0.6, Tu<0.6)
    - CONCEPT: For teaching missing prerequisite concepts

Main Components:
    - graph: The compiled LangGraph workflow
    - schemas: State and response models
    - nodes: Pedagogical node implementations
    - prompts: Comprehensive teaching prompts
    - config: Configuration settings
"""

from educational_agent_math_tutor.graph import graph
from educational_agent_math_tutor.schemas import MathAgentState

__all__ = ["graph", "MathAgentState"]
