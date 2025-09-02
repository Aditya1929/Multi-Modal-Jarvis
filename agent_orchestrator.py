import re
from agent_tools import AGENT_TOOL_REGISTRY

def handle_agentic_suggestion(suggestion: str):
    """
    Parses a suggestion string like 'SUGGESTION: Book a flight to Delhi.'
    and calls the appropriate tool from AGENT_TOOL_REGISTRY.
    """
    suggestion = suggestion.strip()
    # Example: SUGGESTION: Book a flight to Delhi.
    if suggestion.lower().startswith("suggestion:"):
        action_text = suggestion[len("suggestion:"):].strip()
        # Very simple pattern matching for demo purposes
        if "flight to" in action_text.lower():
            # Extract destination
            match = re.search(r"flight to ([A-Za-z ]+)", action_text, re.IGNORECASE)
            if match:
                destination = match.group(1).strip()
                return AGENT_TOOL_REGISTRY["book_flight"](destination)
        elif "send email" in action_text.lower():
            # Example: SUGGESTION: Send email to John about the meeting.
            # (You can expand this pattern as needed)
            return AGENT_TOOL_REGISTRY["send_email"]("John", "Meeting", "Details about the meeting.")
        elif "remind" in action_text.lower():
            # Example: SUGGESTION: Remind me to call mom at 5pm.
            return AGENT_TOOL_REGISTRY["set_reminder"]("call mom", "5pm")
        else:
            return f"[AGENT] No matching tool for suggestion: {action_text}"
    return None 