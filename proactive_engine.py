import assist

def action_get_weather_forecast(feedback_logger):
    """Action to get the weather forecast."""
    print("Proactive action: Getting weather forecast.")
    # In a real implementation, this would call a weather API
    weather_summary = assist.ask_question_memory("What is the weather forecast for today?")
    response = f"Before you go, here is the weather forecast: {weather_summary} Was this helpful?"
    assist.TTS(response)
    # In a real system, you would wait for a "yes" or "no" response here
    # For now, we'll simulate positive feedback for demonstration
    feedback_logger("positive")

def action_suggest_reading_playlist(feedback_logger):
    """Action to suggest a reading playlist."""
    print("Proactive action: Suggesting reading playlist.")
    response = "It looks like you're settling in with a book. Would you like me to play a relaxing reading playlist on Spotify? Was this helpful?"
    assist.TTS(response)
    # Simulate feedback
    feedback_logger("positive")

# --- Proactive Rules ---
# Each rule is a dictionary containing the conditions to check and the action to trigger.
# 'objects': A set of objects that must be detected.
# 'keywords': A set of keywords that must be in the transcript.
# 'action': The function to call if the conditions are met.

PROACTIVE_RULES = [
    {
        "name": "Leaving Home",
        "objects": {"keys", "wallet", "person"},
        "keywords": {"leaving", "heading out", "goodbye", "bye"},
        "action": action_get_weather_forecast
    },
    {
        "name": "Reading Time",
        "objects": {"book"},
        "keywords": {"reading", "read", "settling in"},
        "action": action_suggest_reading_playlist
    }
]

class ProactiveEngine:
    def __init__(self, rules):
        self.rules = rules
        self.cooldowns = {rule['name']: 0 for rule in rules}
        self.COOLDOWN_PERIOD = 300  # 5 minutes in seconds

    def check_triggers(self, detected_objects, transcript, current_time, feedback_logger):
        """Check all proactive rules against the current context."""
        for rule in self.rules:
            # Check if cooldown has passed
            if current_time - self.cooldowns[rule['name']] < self.COOLDOWN_PERIOD:
                continue

            # Check object and keyword conditions
            objects_met = rule['objects'].issubset(detected_objects)
            keywords_met = any(keyword in transcript.lower() for keyword in rule['keywords'])

            if objects_met and keywords_met:
                print(f"Triggering proactive rule: {rule['name']}")
                rule['action'](feedback_logger)
                # Reset cooldown
                self.cooldowns[rule['name']] = current_time
                # Stop after one trigger to avoid multiple actions at once
                return 