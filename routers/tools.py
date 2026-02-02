from langchain_core.tools import tool
from typing import Optional
from .web_helpers import run_web_search
from .emergency import execute_emergency_trigger

@tool
def WebSearchTool(query: str) -> str:
    """
    Performs a web search to find current information, news, or specific facts.
    Use this when the user asks about recent events or topics not in your training data.
    """
    return run_web_search(query)

@tool
def EmergencyCallTool(location: Optional[str] = None) -> str:
    """
    Triggers a CRITICAL Emergency VOICE CALL and SMS to the user's emergency contact.
    ONLY use this if the user expresses immediate intent of SUICIDE ("I will kill myself") or life-threatening danger.
    Do NOT use for general stress or anxiety.
    """
    return execute_emergency_trigger(type="sos", severity="critical", location=location or "Context: Chatbot Trigger")

@tool
def EmergencySmsTool(location: Optional[str] = None) -> str:
    """
    Triggers an Emergency SMS Alert (No Voice Call) to the contacts.
    Use this for MEDIUM severity distress, such as expressions of SELF-HARM ("I might hurt myself") 
    without immediate suicide intent, or if the user asks for help but it's not life-or-death yet.
    """
    return execute_emergency_trigger(type="sos", severity="medium", location=location or "Context: Chatbot Trigger")
