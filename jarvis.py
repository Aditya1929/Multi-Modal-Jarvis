from RealtimeSTT import AudioToTextRecorder
import assist
import time
import tools
import json
import os
from datetime import datetime
from collections import Counter

# Vision log file path
VISION_LOG_FILE = 'jarvis_vision_log.json'
SNAPSHOT_METADATA_FILE = 'snapshot_metadata.json'
LOCATION_SUMMARIES_DIR = 'location_summaries'
VOICE_SUMMARY_FILE = os.path.join('voice_summary.txt')

def load_vision_log():
    """Load the vision log from file"""
    if os.path.exists(VISION_LOG_FILE):
        try:
            with open(VISION_LOG_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def load_snapshot_metadata():
    """Load snapshot metadata from file"""
    if os.path.exists(SNAPSHOT_METADATA_FILE):
        try:
            with open(SNAPSHOT_METADATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def get_todays_objects():
    """Get all objects detected today from vision log"""
    vision_log = load_vision_log()
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get all objects from today
    todays_objects = []
    for entry in vision_log:
        if entry.get('date') == today:
            todays_objects.extend(entry.get('objects', []))
    
    # Count occurrences
    object_counts = Counter(todays_objects)
    
    if not object_counts:
        return "I haven't seen any objects today yet."
    
    # Format the response
    response_parts = []
    for obj, count in object_counts.most_common():
        if count == 1:
            response_parts.append(f"a {obj}")
        else:
            response_parts.append(f"{count} {obj}s")
    
    return f"Today I've seen: {', '.join(response_parts)}."

def get_snapshot_summary(date=None):
    """Get summary of snapshots for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    metadata = load_snapshot_metadata()
    day_snapshots = [entry for entry in metadata if entry.get('date') == date]
    
    if not day_snapshots:
        return f"I didn't take any snapshots on {date}."
    
    # Get unique scene descriptions
    scene_descriptions = [entry.get('scene_label', 'Unknown scene') for entry in day_snapshots]
    scene_counts = Counter(scene_descriptions)
    
    # Create summary
    summary = f"On {date}, I took {len(day_snapshots)} snapshots and captured:\n"
    for scene, count in scene_counts.most_common():
        if count == 1:
            summary += f"- {scene}\n"
        else:
            summary += f"- {scene} ({count} times)\n"
    
    # Add snapshot types
    periodic_count = len([s for s in day_snapshots if s.get('snapshot_type') == 'periodic'])
    change_count = len([s for s in day_snapshots if s.get('snapshot_type') == 'change'])
    
    summary += f"\nSnapshot breakdown: {periodic_count} periodic, {change_count} change-triggered"
    
    return summary

def check_vision_question(text):
    """Check if the question is about vision/objects seen"""
    vision_keywords = ['see', 'saw', 'seen', 'detect', 'object', 'vision', 'camera', 'eye']
    text_lower = text.lower()
    
    return any(keyword in text_lower for keyword in vision_keywords)

def check_snapshot_question(text):
    """Check if the question is about snapshots"""
    snapshot_keywords = ['snapshot', 'photo', 'picture', 'image', 'capture', 'memory', 'timeline', 'scene']
    text_lower = text.lower()
    
    return any(keyword in text_lower for keyword in snapshot_keywords)

def get_brief_location_description():
    # Read all location summaries
    summaries = []
    if os.path.exists(LOCATION_SUMMARIES_DIR):
        for fname in sorted(os.listdir(LOCATION_SUMMARIES_DIR)):
            if fname.endswith('.txt'):
                with open(os.path.join(LOCATION_SUMMARIES_DIR, fname), 'r') as f:
                    summaries.append(f.read().strip())
    if not summaries:
        return "I don't have enough context to summarize your movements."
    # Compose a prompt for the LLM
    prompt = (
        "You are an AI assistant. Here are summaries of what was seen in different locations, in order. "
        "Write a brief, natural language description of what happened, as if describing a person's day or journey. "
        "Be concise and combine the information into a single paragraph.\n\n"
    )
    for i, summary in enumerate(summaries, 1):
        prompt += f"Location {i}: {summary}\n"
    prompt += "\nBrief description: "
    # Use the LLM to generate the description
    return assist.ask_question_memory(prompt)

def get_voice_summary():
    if os.path.exists(VOICE_SUMMARY_FILE):
        with open(VOICE_SUMMARY_FILE, 'r') as f:
            summary = f.read().strip()
        if summary:
            return summary
    return "I don't have a voice conversation summary yet."

if __name__ == '__main__':
    recorder = AudioToTextRecorder(spinner=False, model="tiny.en", language="en", post_speech_silence_duration =0.1, silero_sensitivity = 0.4)
    hot_words = ["jarvis"]
    skip_hot_word_check = False
    print("Say something...")
    while True:
        current_text = recorder.text()
        print(current_text)
        if any(hot_word in current_text.lower() for hot_word in hot_words) or skip_hot_word_check:
                    #make sure there is text
                    if current_text:
                        print("User: " + current_text)
                        recorder.stop()
                        
                        # Check if it's a snapshot-related question
                        if check_snapshot_question(current_text):
                            snapshot_response = get_snapshot_summary()
                            print("Snapshot Response: " + snapshot_response)
                            assist.TTS(snapshot_response)
                        # Check if it's a vision-related question
                        elif check_vision_question(current_text):
                            vision_response = get_todays_objects()
                            print("Vision Response: " + vision_response)
                            assist.TTS(vision_response)
                        # Check if it's a brief description request
                        elif "brief description" in current_text.lower() or "summarize my day" in current_text.lower():
                            brief_desc = get_brief_location_description()
                            print("Brief Description: " + brief_desc)
                            assist.TTS(brief_desc)
                        # Check if it's a voice summary request
                        elif "voice summary" in current_text.lower() or "conversation summary" in current_text.lower():
                            voice_summary = get_voice_summary()
                            print("Voice Summary: " + voice_summary)
                            assist.TTS(voice_summary)
                        else:
                            #get time
                            current_text = current_text + " " + time.strftime("%Y-m-%d %H-%M-%S")
                            response = assist.ask_question_memory(current_text)
                            print(response)
                            speech = response.split('#')[0]
                            done = assist.TTS(speech)
                            skip_hot_word_check = True if "?" in response else False
                            if len(response.split('#')) > 1:
                                command = response.split('#')[1]
                                tools.parse_command(command)
                        recorder.start()
