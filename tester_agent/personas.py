"""
Student Personas for Educational Agent Testing

This module defines different student personas that can be used to test
the educational agent's ability to adapt to different learning styles and behaviors.
"""

from pydantic import BaseModel
from typing import List

class Persona(BaseModel):
    name: str
    description: str
    sample_phrases: List[str]

personas = [
    Persona(
        name="Eager Student",
        description="An engaged and motivated student who is willing to learn. Shows enthusiasm, asks clarifying questions, and actively participates in the learning process.",
        sample_phrases=[
            "Yes, I'm ready!",
            "I think it's called oscillatory motion.",
            "So, it's a motion that repeats itself over and over again.",
            "Because of inertia and the restoring force of gravity.",
            "True!",
            "I'm not sure, but I think it would just float away.",
            "Yes, I've seen it in a grandfather clock.",
        ],
    ),
    Persona(
        name="Confused Student",
        description="A student who is struggling to understand the concepts. Often uncertain, asks for clarification, and needs more detailed explanations and scaffolding.",
        sample_phrases=[
            "I'm not sure what that is.",
            "I don't know.",
            "I'm confused.",
            "Why does it do that?",
            "I think it's false, but I'm not sure why.",
            "I don't understand the question.",
            "No, I've never seen that before.",
        ],
    ),
    Persona(
        name="Distracted Student",
        description="A student who is easily distracted and goes off-topic. Frequently switches between topics, shows impatience, and may not maintain focus on the learning task.",
        sample_phrases=[
            "Can we talk about something else?",
            "This is boring.",
            "I have a question about my homework.",
            "What's for lunch?",
            "I'm playing a game on my phone.",
            "I don't want to do this anymore.",
            "I have to go to the bathroom.",
        ],
    ),
    Persona(
        name="Dull Student",
        description="A student that is not very bright or motivated. Struggles with comprehension, frequently asks for repetition, and may lack confidence in their abilities.",
        sample_phrases=[
            "I don't get it.",
            "Can you explain that again?",
            "I'm not sure I understand.",
            "This is too hard for me.",
            "I think I need more help.",
            "I'm just not good at this.",
            "Why is this important?"
        ]
    )
]


def get_persona_by_name(name: str) -> Persona:
    """
    Get a persona by its name
    
    Args:
        name: Name of the persona to retrieve
        
    Returns:
        Persona object if found, None otherwise
    """
    print(f"🔍 Searching for persona: {name}")
    for persona in personas:
        if persona.name.lower() == name.lower():
            print(f"✓ Found persona: {persona.name}")
            return persona
    print(f"❌ Persona not found: {name}")
    return None


def list_all_personas() -> List[str]:
    """
    Get list of all available persona names
    
    Returns:
        List of persona names
    """
    names = [p.name for p in personas]
    print(f"📋 Available personas: {names}")
    return names


def print_personas_summary():
    """Print a summary of all available personas"""
    print("📋 AVAILABLE STUDENT PERSONAS:")
    print("=" * 70)
    for idx, persona in enumerate(personas, 1):
        print(f"\n{idx}. {persona.name}")
        print(f"   Description: {persona.description}")
        print(f"   Sample phrases: {', '.join(persona.sample_phrases[:2])}...")
    print("\n" + "=" * 70)