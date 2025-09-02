import json
from datetime import datetime
import os

LOGS_DIR = 'multimodal_logs'
VISION_LOG_FILE = os.path.join(LOGS_DIR, 'vision_context_log.json')
VOICE_LOG_FILE = os.path.join(LOGS_DIR, 'voice_context_log.json')

def load_log(file_path):
    """Loads a JSON log file."""
    log = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                log.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")
    return log

def correlate_contexts(time_window_seconds=5):
    """Correlates vision and voice contexts within a time window."""
    vision_log = load_log(VISION_LOG_FILE)
    voice_log = load_log(VOICE_LOG_FILE)

    if not vision_log or not voice_log:
        print("Not enough data to correlate.")
        return

    # Sort logs by timestamp
    vision_log.sort(key=lambda x: x['timestamp'])
    voice_log.sort(key=lambda x: x['timestamp'])

    print("--- Context Correlation Report ---")

    for vision_entry in vision_log:
        vision_time = datetime.fromtimestamp(vision_entry['timestamp'])
        
        # Find nearby voice entries
        nearby_voice_entries = []
        for voice_entry in voice_log:
            voice_time = datetime.fromtimestamp(voice_entry['timestamp'])
            time_diff = abs((vision_time - voice_time).total_seconds())
            if time_diff <= time_window_seconds:
                nearby_voice_entries.append(voice_entry)

        if nearby_voice_entries:
            print(f"\nAt {vision_time.strftime('%Y-%m-%d %H:%M:%S')}, I saw: {vision_entry['caption']} (Objects: {', '.join(vision_entry['objects'])})")
            for voice_entry in nearby_voice_entries:
                voice_time = datetime.fromtimestamp(voice_entry['timestamp'])
                print(f"  - Around the same time ({voice_time.strftime('%H:%M:%S')}), I heard: \"{voice_entry['transcript']}\"")

if __name__ == '__main__':
    correlate_contexts() 