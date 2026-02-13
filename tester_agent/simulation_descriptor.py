"""
Simple Simulation Descriptor for Tester Agent

This module provides straightforward descriptions of what a person would observe
in a pendulum simulation, without complex timing or phase tracking.
"""

from typing import Dict, Any

def describe_simulation_for_tester(simulation_config: Dict[str, Any]) -> str:
    """
    Generate a simple description of what a person would observe in the pendulum simulation.
    No complex timing or phases - just straightforward observations.
    """
    if not simulation_config:
        return ""
    
    concept = simulation_config.get("concept")
    before_params = simulation_config.get("before_params")
    after_params = simulation_config.get("after_params")
    action_description = simulation_config.get("action_description")
    agent_message = simulation_config.get("agent_message")
    
    # What changes and what you'd observe
    changes_description = _describe_parameter_change(before_params, after_params)
    observation_description = _describe_visual_observations(before_params, after_params)
    
    description = f"""
🔬 **SIMULATION: {concept}**

The teacher is showing you a pendulum simulation. Here's what you can observe:

**What's happening:** {action_description}

**The setup:**
- Initial pendulum: Length={before_params.get('length', '?')}m, Gravity={before_params.get('gravity', '?')} m/s², Starting angle={before_params.get('amplitude', '?')}°
- Modified pendulum: Length={after_params.get('length', '?')}m, Gravity={after_params.get('gravity', '?')} m/s², Starting angle={after_params.get('amplitude', '?')}°

**What you would observe:**
{observation_description}

**Key insight:** {changes_description}

The teacher said: "{agent_message}"
    """.strip()
    
    return description

def _describe_parameter_change(before_params: Dict, after_params: Dict) -> str:
    """Simple description of what parameter changed and its effect."""
    
    length_before = before_params.get('length', 0)
    length_after = after_params.get('length', 0)
    gravity_before = before_params.get('gravity', 0)
    gravity_after = after_params.get('gravity', 0)
    amplitude_before = before_params.get('amplitude', 0)
    amplitude_after = after_params.get('amplitude', 0)
    
    if length_before != length_after:
        if length_after > length_before:
            return f"Making the pendulum longer (from {length_before}m to {length_after}m) makes it swing slower - longer period"
        else:
            return f"Making the pendulum shorter (from {length_before}m to {length_after}m) makes it swing faster - shorter period"
    
    elif gravity_before != gravity_after:
        if gravity_after > gravity_before:
            return f"Increasing gravity (from {gravity_before} to {gravity_after} m/s²) makes the pendulum swing faster"
        else:
            return f"Decreasing gravity (from {gravity_before} to {gravity_after} m/s²) makes the pendulum swing slower"
    
    elif amplitude_before != amplitude_after:
        return f"Changing the starting angle (from {amplitude_before}° to {amplitude_after}°) shows that amplitude doesn't significantly affect the period for small angles"
    
    else:
        return "The parameters stay the same, demonstrating that some factors (like mass) don't affect the pendulum's period"

def _describe_visual_observations(before_params: Dict, after_params: Dict) -> str:
    """Simple description of what you would visually observe."""
    
    length_before = before_params.get('length')
    length_after = after_params.get('length')
    gravity_before = before_params.get('gravity')
    gravity_after = after_params.get('gravity')
    amplitude_before = before_params.get('amplitude')
    amplitude_after = after_params.get('amplitude')
    
    observations = []
    observations.append("- You see a pendulum swinging back and forth")
    observations.append("- First, it swings with the initial settings")
    observations.append("- Then the settings change and you can compare the difference")
    
    if length_before != length_after:
        if length_after > length_before:
            observations.append("- The longer pendulum takes more time for each complete swing")
            observations.append("- You can count and compare: longer pendulum = slower swings")
        else:
            observations.append("- The shorter pendulum completes swings faster")
            observations.append("- You can count and compare: shorter pendulum = quicker swings")
    
    elif gravity_before != gravity_after:
        if gravity_after > gravity_before:
            observations.append("- With higher gravity, the pendulum swings much faster")
            observations.append("- The stronger pull makes it swing back and forth more quickly")
        else:
            observations.append("- With lower gravity, the pendulum swings more slowly")
            observations.append("- The weaker pull makes it take longer for each swing")
    
    elif amplitude_before != amplitude_after:
        observations.append("- Even with different starting angles, the timing stays roughly the same")
        observations.append("- This shows that how far you pull it back doesn't change how fast it swings")
    
    return "\n".join(observations)


# Main integration functions - SIMPLIFIED
def get_simulation_description(agent_state: Dict[str, Any]) -> str:
    """
    Get simple simulation description for tester agent.
    Just describes what they would observe - no complex timing or phases.
    """
    simulation_config = agent_state.get("simulation_config", {})
    
    # Simple check: if there's a simulation config, describe it
    if simulation_config:
        return describe_simulation_for_tester(simulation_config)
    
    return ""

def format_simulation_context_for_tester(agent_state: Dict[str, Any]) -> str:
    """
    Format simulation information specifically for the tester agent's context.
    """
    simulation_description = get_simulation_description(agent_state)
    
    # No redundant check needed - if description is empty, return empty
    if simulation_description:
        return f"""
📺 **SIMULATION HAPPENING RIGHT NOW:**

{simulation_description}

**For your response:**
- Respond as if you're watching this simulation
- Comment on what you observe  
- Ask questions if curious about what you see
- Stay in character while discussing the pendulum
"""
    
    return ""
