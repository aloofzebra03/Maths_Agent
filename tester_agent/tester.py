import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from tester_agent.personas import Persona
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_groq import ChatGroq

class TesterAgent:
    def __init__(self, persona: Persona, is_kannada: bool = False):
        self.persona = persona
        self.is_kannada = is_kannada
        self.response_count = 0
        
        print(f"🎭 Initializing TesterAgent with persona: {persona.name}")
        print(f"   Description: {persona.description}")
        print(f"   Kannada mode: {is_kannada}")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-3-27b-it",
            api_key=os.getenv("GOOGLE_API_KEY_2"),
            temperature=0.7,
        )
        print("✓ LLM initialized")
    
        # self.llm = ChatGroq(
        # model="llama-3.3-70b-versatile",
        # temperature=0.5,
        # max_tokens=None,
    # )
        # Initialize history with the persona's description as a system prompt
        base_prompt = f"You are a student with the persona of a '{self.persona.name}'. Your characteristics are: {self.persona.description}. You must consistently act according to this persona."
        
        # Add Kannada instruction if is_kannada is True
        if self.is_kannada:
            base_prompt += " You are a student of Kannada origin and understand only Kannada. You must respond ONLY in Kannada language. All your responses must be in Kannada script, not English."
        
        self.history = [
            SystemMessage(content=base_prompt)
        ]
        print(f"✓ System prompt set for {persona.name}")

    def respond(self, agent_msg: str) -> str:
        """Generate a student response based on agent message"""
        self.response_count += 1
        print(f"\n📨 Response #{self.response_count} generation started")
        print(f"   Input from agent: {agent_msg[:80]}..." if len(agent_msg) > 80 else f"   Input from agent: {agent_msg}")
        
        # Add the educational agent's latest message to the history
        self.history.append(AIMessage(content=agent_msg))
        print(f"   History size: {len(self.history)} messages")

        formatted_history = self.history

        base_prompt = f"""
{formatted_history}

Your task is to provide the next response as the "User", staying true to your persona.
Your response should be on a single line.

User: """
        
        if self.is_kannada:
            base_prompt += " You are a student of Kannada origin and understand only Kannada. You must respond ONLY in Kannada language. All your responses must be in Kannada script, not English."

        # Generate response using the context-rich prompt
        print("   → Calling LLM for response generation...")
        response = self.llm.invoke(base_prompt)
        user_response = response.content.strip()
        
        print(f"   ✓ Response generated: {user_response[:80]}..." if len(user_response) > 80 else f"   ✓ Response generated: {user_response}")

        # Add your own response to the history
        self.history.append(HumanMessage(content=user_response))

        return user_response

    def respond_with_simulation_context(self, agent_msg: str, simulation_description: str) -> str:
        """
        Respond to the agent message with additional context about a visual simulation.
        This method provides the tester agent with textual description of what's happening
        in the simulation so it can respond appropriately.
        """
        self.response_count += 1
        print(f"\n📨 Simulation-context response #{self.response_count} generation started")
        print(f"   Input from agent: {agent_msg[:80]}..." if len(agent_msg) > 80 else f"   Input from agent: {agent_msg}")
        print(f"   Simulation context provided: {simulation_description[:80]}...")
        
        # Add the educational agent's latest message to the history
        self.history.append(AIMessage(content=agent_msg))
        print(f"   History size: {len(self.history)} messages")

        formatted_history = self.history

        prompt = f"""
{formatted_history}

{simulation_description}

Your task is to provide the next response as the "User", staying true to your persona.
Based on the simulation description provided above, respond as if you can see and observe what's happening in the simulation.
Your response should acknowledge or comment on the simulation if relevant.

User: """

        # Generate response using the context-rich prompt with simulation context
        print("   → Calling LLM for response generation with simulation context...")
        response = self.llm.invoke(prompt)
        user_response = response.content.strip()
        
        print(f"   ✓ Response generated: {user_response[:80]}..." if len(user_response) > 80 else f"   ✓ Response generated: {user_response}")

        # Add your own response to the history
        self.history.append(HumanMessage(content=user_response))

        return user_response