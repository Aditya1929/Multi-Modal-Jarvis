def book_flight(destination):
    print(f"[AGENT TOOL] Booking a flight to {destination}...")
    return f"Flight to {destination} booked!"

def send_email(recipient, subject, body):
    print(f"[AGENT TOOL] Sending email to {recipient} with subject '{subject}'...")
    return f"Email sent to {recipient} with subject '{subject}'."

def set_reminder(what, when):
    print(f"[AGENT TOOL] Setting reminder: {what} at {when}")
    return f"Reminder set: {what} at {when}."

# Registry for dynamic lookup
AGENT_TOOL_REGISTRY = {
    "book_flight": book_flight,
    "send_email": send_email,
    "set_reminder": set_reminder,
} 