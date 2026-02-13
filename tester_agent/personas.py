from pydantic import BaseModel
from typing import List

class Persona(BaseModel):
    name: str
    description: str
    sample_phrases: List[str]

personas = [
    Persona(
        name="Eager Student",
        description="An engaged and motivated student who is willing to learn.",
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
        description="A student who is struggling to understand the concepts.",
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
        description="A student who is easily distracted and goes off-topic.",
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
        description="A student that is not very bright.",
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