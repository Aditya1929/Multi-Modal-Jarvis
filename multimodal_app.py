import pygame
import sys
import cv2
import numpy as np
import os
from dotenv import load_dotenv
from camera_manager import CameraManager
import json
from datetime import datetime
import time
from PIL import Image, ImageDraw, ImageFont
import hashlib
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from ultralytics import YOLO
from collections import deque
import clip
from scipy.spatial.distance import cosine
from RealtimeSTT import AudioToTextRecorder
import assist
from proactive_engine import ProactiveEngine, PROACTIVE_RULES
from memory_manager import add_memory, search_memories
from agent_orchestrator import handle_agentic_suggestion
import glob

load_dotenv()

# Initialize Pygame
pygame.init()

# --- Constants ---
SCREEN_WIDTH = int(os.getenv('SCREEN_WIDTH', '800'))
SCREEN_HEIGHT = int(os.getenv('SCREEN_HEIGHT', '600'))
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
NAVY_BLUE = (20, 20, 40)
LIGHT_BLUE = (173, 216, 230)
GRAY = (40, 40, 40)

FONT = pygame.font.Font(None, 32)
SMALL_FONT = pygame.font.Font(None, 24)

# --- Directories and files ---
LOGS_DIR = 'multimodal_logs'
VISION_LOG_FILE = os.path.join(LOGS_DIR, 'vision_context_log.json')
VOICE_LOG_FILE = os.path.join(LOGS_DIR, 'voice_context_log.json')
VOICE_SUMMARY_FILE = os.path.join(LOGS_DIR, 'voice_summary.txt')
LOCATION_SUMMARY_FILE = os.path.join(LOGS_DIR, 'location_current.txt')
SNAPSHOT_METADATA_FILE = os.path.join(LOGS_DIR, 'snapshot_metadata.json')
COMPILED_LOCATION_CONTEXT_FILE = os.path.join(LOGS_DIR, 'compiled_location_context.txt')
EVENT_LOG_FILE = os.path.join(LOGS_DIR, 'event_log.json')

# --- Model Initialization ---
print("Loading image captioning model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP model loaded successfully!")

print("Loading YOLOv8 model...")
yolo_model = YOLO('yolov8s.pt')
print("YOLOv8 model loaded successfully!")

CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=CLIP_DEVICE)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_log_dirs():
    ensure_dir(LOGS_DIR)

def label_image_with_ai(image):
    """Label image using Hugging Face BLIP model"""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        inputs = processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_length=50, num_beams=5)
            caption = processor.decode(out[0], skip_special_tokens=True)
        caption = caption.strip()
        if not caption:
            caption = "A scene captured by the camera"
        return caption
    except Exception as e:
        print(f"Error in AI labeling: {e}")
        return "A scene captured by the camera"

def detect_objects_yolo(image):
    """Detect objects using YOLOv8 model"""
    results = yolo_model(image, verbose=False)[0]
    detected_objects = set()
    for box in results.boxes:
        class_id = int(box.cls)
        label = yolo_model.names[class_id]
        detected_objects.add(label)
    return detected_objects

def get_scene_embedding(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = clip_preprocess(image).unsqueeze(0).to(CLIP_DEVICE)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()

#def check_vision_question(text):
    """Check if the question is about vision/objects seen"""
    vision_keywords = ['see', 'saw', 'seen', 'detect', 'object', 'vision', 'camera', 'eye']
    text_lower = text.lower()
    
    return any(keyword in text_lower for keyword in vision_keywords)

def get_todays_objects(context_buffer=None):
    """Get all objects detected today from vision log"""
    # Load vision log
    vision_log = []
    if os.path.exists(VISION_LOG_FILE):
        with open(VISION_LOG_FILE, 'r') as f:
            vision_log = json.load(f)

    # Get all objects from today
    today = datetime.now().strftime('%Y-%m-%d')
    todays_objects = []
    for entry in vision_log:
        entry_date = datetime.fromtimestamp(entry.get('timestamp')).strftime('%Y-%m-%d')
        if entry_date == today:
            todays_objects.extend(entry.get('objects', []))
    
    # Count occurrences
    from collections import Counter
    object_counts = Counter(todays_objects)
    
    if not object_counts:
        return "I haven't seen any objects recently."
    
    # Format the response
    response_parts = []
    for obj, count in object_counts.most_common():
        if count == 1:
            response_parts.append(f"a {obj}")
        else:
            response_parts.append(f"{count} {obj}s")
    
    return f"I've recently seen: {', '.join(response_parts)}."

def summarize_voice_context(voice_buffer):
    """Generates and saves a summary of the voice conversation and stores it in memory DB."""
    if not voice_buffer:
        return "No voice context available."

    # Create a prompt for the LLM
    prompt = "You are an AI assistant. Based on the following conversation transcript, write a concise, two-sentence summary of the key topics discussed.\n\n"
    
    full_transcript = "\n".join(voice_buffer)
    prompt += "Transcript:\n" + full_transcript + "\n\n"
    prompt += "Summary of conversation:"

    print("Generating voice summary...")
    summary = assist.ask_question_memory(prompt)

    # Save the summary, overwriting the old one
    with open(VOICE_SUMMARY_FILE, 'w') as f:
        f.write(summary)
    print(f"Saved voice summary to {VOICE_SUMMARY_FILE}")

    # Store in memory DB
    import clip
    import torch
    text_tokens = clip.tokenize([summary]).to(CLIP_DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    embedding = text_features.cpu().numpy().flatten()
    add_memory(embedding, {"type": "voice_summary", "summary": summary, "timestamp": time.time()})

    return summary if summary else "No voice context available."

def load_snapshot_metadata():
    """Load snapshot metadata from file"""
    if os.path.exists(SNAPSHOT_METADATA_FILE):
        try:
            with open(SNAPSHOT_METADATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

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
    from collections import Counter
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
    
    return summary if summary else f"I didn't take any snapshots on {date}."

def check_snapshot_question(text):
    """Check if the question is about snapshots"""
    snapshot_keywords = ['snapshot', 'photo', 'picture', 'image', 'capture', 'memory', 'timeline', 'scene']
    text_lower = text.lower()
    
    return any(keyword in text_lower for keyword in snapshot_keywords)

def update_location_summary(context_buffer):
    """Update the current location summary file with the latest scene captions."""
    if context_buffer:
        # Use the last N captions for a brief summary
        N = 5
        captions = [entry['caption'] for entry in list(context_buffer)[-N:]]
        summary = '\n'.join(captions)
        with open(LOCATION_SUMMARY_FILE, 'w') as f:
            f.write(summary)

def summarize_location_context(location_buffer, location_id):
    """Generates and saves a summary for a specific location's context and stores it in memory DB."""
    if not location_buffer:
        return

    # Create a directory for the location
    location_dir = os.path.join(LOGS_DIR, f'location_{location_id}')
    ensure_dir(location_dir)

    # Create a prompt for the LLM
    prompt = "You are an AI assistant. Based on the following observations, write a concise, one-paragraph summary of what happened at this location.\n\n"
    all_captions = [entry.get('caption', '') for entry in location_buffer]
    all_objects = set()
    for entry in location_buffer:
        for obj in entry.get('objects', []):
            all_objects.add(obj)
    
    prompt += "Observed scenes:\n" + "\n".join(f"- {c}" for c in all_captions) + "\n\n"
    prompt += "Detected objects: " + ", ".join(list(all_objects)) + "\n\n"
    prompt += "Summary of location:"

    print(f"Generating summary for location {location_id}...")
    summary = assist.ask_question_memory(prompt)

    # Save the summary to its own file in its own directory
    summary_filename = os.path.join(location_dir, 'summary.txt')
    with open(summary_filename, 'w') as f:
        f.write(summary)
    print(f"Saved summary to {summary_filename}")

    # Store in memory DB
    import clip
    import torch
    text_tokens = clip.tokenize([summary]).to(CLIP_DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    embedding = text_features.cpu().numpy().flatten()
    add_memory(embedding, {"type": "location_summary", "summary": summary, "location_id": str(location_id), "timestamp": time.time()})

def update_compiled_context(max_location_id):
    """Reads all location summaries and compiles them into a single, narrative summary."""
    all_summaries = []
    for i in range(1, max_location_id + 1):
        summary_filename = os.path.join(LOGS_DIR, f'location_{i}', 'summary.txt')
        if os.path.exists(summary_filename):
            with open(summary_filename, 'r') as f:
                all_summaries.append(f"Location {i}: {f.read().strip()}")

    if not all_summaries:
        return

    # Create a prompt for the LLM to generate a single compiled summary
    prompt = "You are an AI assistant. Based on the following summaries from different locations, write a single, cohesive, narrative summary of the day's journey. Combine the events smoothly into one paragraph.\n\n"
    prompt += "\n".join(all_summaries)
    prompt += "\n\nCompiled narrative of the day:"

    print("Generating compiled location summary...")
    compiled_summary = assist.ask_question_memory(prompt)

    with open(COMPILED_LOCATION_CONTEXT_FILE, 'w') as compiled_file:
        compiled_file.write(compiled_summary)
        
    print(f"Updated compiled context file: {COMPILED_LOCATION_CONTEXT_FILE}")

def log_event(event_type, context, feedback=None):
    """Logs an event (proactive action or user command) for learning."""
    # Convert sets in context to lists for JSON serialization
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(i) for i in obj]
        else:
            return obj
    context = convert_sets(context)
    event_data = {
        "timestamp": time.time(),
        "event_type": event_type,
        "context": context,
        "feedback": feedback
    }
    
    # Load existing log and append
    try:
        with open(EVENT_LOG_FILE, 'r+') as f:
            log = json.load(f)
            log.append(event_data)
            f.seek(0)
            json.dump(log, f, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(EVENT_LOG_FILE, 'w') as f:
            json.dump([event_data], f, indent=4)

def get_brief_location_description():
    """Reads and returns the compiled location context file."""
    if os.path.exists(COMPILED_LOCATION_CONTEXT_FILE):
        with open(COMPILED_LOCATION_CONTEXT_FILE, 'r') as f:
            summary = f.read().strip()
        if summary:
            return summary
    return "No location summary available."

def get_brief_location_description_today():
    """Summarize only today's location summaries."""
    today = datetime.now().strftime('%Y-%m-%d')
    summaries = []
    for summary_filename in glob.glob(os.path.join(LOGS_DIR, 'location_*', 'summary.txt')):
        ts = datetime.fromtimestamp(os.path.getmtime(summary_filename)).strftime('%Y-%m-%d')
        if ts == today:
            loc = os.path.basename(os.path.dirname(summary_filename))
            with open(summary_filename, 'r') as f:
                summaries.append(f"{loc}: {f.read().strip()}")
    if not summaries:
        return "No location summaries for today."
    prompt = f"Summarize the following locations for today:\n" + "\n".join(summaries)
    return assist.ask_question_memory(prompt)

def get_full_location_description():
    """Summarize all location summaries (all time)."""
    summaries = []
    for i in range(1, 1000):
        summary_filename = os.path.join(LOGS_DIR, f'location_{i}', 'summary.txt')
        if not os.path.exists(summary_filename):
            break
        with open(summary_filename, 'r') as f:
            summaries.append(f"Location {i}: {f.read().strip()}")
    if not summaries:
        return "No location summaries available."
    prompt = f"Summarize the following locations:\n" + "\n".join(summaries)
    return assist.ask_question_memory(prompt)

def get_current_scene_caption(current_scene_label):
    """Return the caption of the current frame."""
    return current_scene_label if current_scene_label else "No scene detected."

# --- Layout ---
# The screen will be divided into three main sections
# 1. Top-left: Scene Capture (video feed)
# 2. Top-right: Audio Feed (live transcript, context)
# 3. Bottom: Jarvis Transcription (conversation history)

scene_capture_rect = pygame.Rect(0, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
audio_feed_rect = pygame.Rect(SCREEN_WIDTH // 2, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
jarvis_transcription_rect = pygame.Rect(0, SCREEN_HEIGHT // 2, SCREEN_WIDTH, SCREEN_HEIGHT // 2)

def draw_section_title(screen, text, rect):
    """Draws a title for a section."""
    title_surface = FONT.render(text, True, LIGHT_BLUE)
    title_rect = title_surface.get_rect(center=(rect.centerx, rect.top + 20))
    screen.blit(title_surface, title_rect)

def run():
    """Main loop for the multimodal application."""
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Holomat Multimodal AI")
    
    # --- Initialization ---
    camera_manager = CameraManager('./M.npy', SCREEN_WIDTH, SCREEN_HEIGHT)
    ensure_log_dirs()

    running = True
    clock = pygame.time.Clock()

    # Context memory
    CONTEXT_SECONDS = 10
    LOCATION_CHANGE_THRESHOLD = 0.28
    context_buffer = deque()
    last_location_embedding = None
    current_scene_label = "Analyzing scene..."
    current_objects = set()
    current_location_id = 1
    location_context_buffer = []


    # Proactive Engine
    proactive_engine = ProactiveEngine(PROACTIVE_RULES)

    # Context for Learning
    recent_context_buffer = deque(maxlen=10) # 10 seconds of context

    # Audio Feed
    recorder = AudioToTextRecorder(spinner=False, model="tiny.en", language="en", post_speech_silence_duration=0.5, silero_sensitivity=0.4)
    recorder.start()
    live_transcript = ""

    # Voice Context
    voice_context_buffer = deque(maxlen=100) # store last 20 transcripts
    last_voice_summary_time = time.time()
    VOICE_SUMMARY_INTERVAL = 60 # seconds

    # Jarvis
    jarvis_history = deque(maxlen=10)
    hot_words = ["jarvis"]
    skip_hot_word_check = False

    while running:
        if not camera_manager.update():
            continue
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Scene Capture Logic ---
        ret, frame = camera_manager.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_transformed = cv2.warpPerspective(frame_rgb, camera_manager.M, (SCREEN_WIDTH, SCREEN_HEIGHT))

            current_time = time.time()

            # Update recent context buffer for learning
            recent_context_buffer.append({
                "timestamp": current_time,
                "objects": current_objects,
                "transcript": live_transcript
            })
            
            scene_embedding = get_scene_embedding(frame_transformed)
            # Store vision memory
            add_memory(scene_embedding, {
                "type": "vision",
                "caption": current_scene_label,
                "objects": ", ".join(str(obj) for obj in list(current_objects)),  # Ensure string
                "timestamp": current_time
            })

            if last_location_embedding is not None:
                dist = cosine(scene_embedding, last_location_embedding)
                if dist > LOCATION_CHANGE_THRESHOLD:
                    print(f"Location changed! Finalizing location {current_location_id}.")
                    summarize_location_context(location_context_buffer, current_location_id)
                    update_compiled_context(current_location_id)
                    
                    current_location_id += 1
                    location_context_buffer.clear()
                    print(f"Started new location: {current_location_id}")

            last_location_embedding = scene_embedding

            if len(context_buffer) == 0 or (current_time - context_buffer[-1]["timestamp"] > 1):
                caption = label_image_with_ai(frame_transformed)
                objects = detect_objects_yolo(frame_transformed)
                entry = {
                    "timestamp": current_time,
                    "caption": caption,
                    "objects": list(objects)
                }
                context_buffer.append(entry)
                location_context_buffer.append(entry)
                with open(VISION_LOG_FILE, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
                update_location_summary(context_buffer)

                while context_buffer and current_time - context_buffer[0]["timestamp"] > CONTEXT_SECONDS:
                    context_buffer.popleft()
                current_scene_label = caption
                current_objects = objects

            # --- Drawing ---
            screen.fill(BLACK)

            # Draw Scene Capture
            scene_surface = pygame.surfarray.make_surface(frame_transformed.transpose((1, 0, 2)))
            scene_surface = pygame.transform.scale(scene_surface, (scene_capture_rect.width, scene_capture_rect.height))
            screen.blit(scene_surface, scene_capture_rect.topleft)
            
            # Draw scene label
            label_surface = SMALL_FONT.render(f"AI: {current_scene_label}", True, WHITE)
            screen.blit(label_surface, (scene_capture_rect.left + 10, scene_capture_rect.bottom - 60))

            # Draw detected objects
            objects_surface = SMALL_FONT.render(f"Objects: {', '.join(current_objects)}", True, LIGHT_BLUE)
            screen.blit(objects_surface, (scene_capture_rect.left + 10, scene_capture_rect.bottom - 30))

        # --- Audio Feed Logic ---
        new_text = recorder.text()
        if new_text:
            live_transcript = new_text
            entry = {
                "timestamp": time.time(),
                "transcript": new_text
            }
            with open(VOICE_LOG_FILE, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            
            voice_context_buffer.append(new_text)

            # Check if it's time to update the voice summary
            current_time = time.time()
            if current_time - last_voice_summary_time > VOICE_SUMMARY_INTERVAL:
                summarize_voice_context(list(voice_context_buffer))
                last_voice_summary_time = current_time

            # --- Proactive Engine Check ---
            if not any(hot_word in new_text.lower() for hot_word in hot_words):
                 # Pass a lambda to log feedback
                 feedback_logger = lambda feedback: log_event('proactive_action', list(recent_context_buffer), feedback)
                 proactive_engine.check_triggers(current_objects, new_text, current_time, feedback_logger)

            if any(hot_word in new_text.lower() for hot_word in hot_words) or skip_hot_word_check:
                if new_text:
                    # Log the user's command as a learning event
                    log_event('user_command', list(recent_context_buffer))
                    jarvis_history.append(f"You: {new_text}")
                    recorder.stop()
                    
                    # --- Jarvis Query Routing ---
                        # If the user asks 'what do you see', just return the current frame caption
                    if "what do you see" in new_text.lower():
                            response = get_current_scene_caption(current_scene_label)
                            
                    elif check_snapshot_question(new_text):
                        response = get_snapshot_summary()
                    elif "voice summary" in new_text.lower() or "conversation summary" in new_text.lower():
                        response = summarize_voice_context(list(voice_context_buffer))
                    elif "brief description" in new_text.lower() or "location summary" in new_text.lower() or "summarize my day" in new_text.lower():
                        # If the user asks for 'today', use the today-only function
                        if "today" in new_text.lower():
                            response = get_brief_location_description_today()
                        else:
                            response = get_full_location_description()
                    else:
                        # --- Fully Context-Aware Querying ---
                        # Aggregate all context
                        # 1. Location summaries
                        location_summaries = []
                        for i in range(1, current_location_id + 1):
                            summary_filename = os.path.join(LOGS_DIR, f'location_{i}', 'summary.txt')
                            if os.path.exists(summary_filename):
                                with open(summary_filename, 'r') as f:
                                    location_summaries.append(f"Location {i}: {f.read().strip()}")
                        location_summaries_str = "\n".join(location_summaries)
                        # 2. Vision log
                        vision_log_str = ""
                        if os.path.exists(VISION_LOG_FILE):
                            with open(VISION_LOG_FILE, 'r') as f:
                                try:
                                    vision_entries = json.load(f)
                                    vision_log_str = "\n".join([
                                        f"[{datetime.fromtimestamp(e['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}] {e['caption']} (Objects: {', '.join(e['objects'])})" for e in vision_entries
                                    ])
                                except Exception:
                                    vision_log_str = ""
                        # 3. Voice summaries
                        voice_summary_str = ""
                        if os.path.exists(VOICE_SUMMARY_FILE):
                            with open(VOICE_SUMMARY_FILE, 'r') as f:
                                voice_summary_str = f.read().strip()
                        # 4. Snapshots
                        snapshot_summary_str = get_snapshot_summary()
                        # Build prompt
                        prompt = f"""
You are an AI assistant with access to the following context data:

Location summaries:
{location_summaries_str}

Vision log:
{vision_log_str}

Voice summary:
{voice_summary_str}

Snapshot summary:
{snapshot_summary_str}

User question: {new_text}

Instructions:
- Use only the relevant data to answer the user's question.
- If the question refers to a specific time, place, person, or event, filter the context accordingly.
- If the question is general, summarize or synthesize as needed.
- If the user's question is agentic (i.e., they want to do something, or you can help them take an action), after your answer, suggest an action in the format:
  SUGGESTION: <action>
  For example: SUGGESTION: Would you like me to book a flight to Delhi?
- If no action is appropriate, do not include a suggestion.
- Be concise and helpful.
"""
                        response = assist.ask_question_memory(prompt)
                        # Parse for SUGGESTION
                        suggestion = None
                        answer = response
                        if response and "SUGGESTION:" in response:
                            parts = response.split("SUGGESTION:", 1)
                            answer = parts[0].strip()
                            suggestion = parts[1].strip()
                        jarvis_history.append(f"Jarvis: {answer}")
                        # If there is a suggestion, handle it
                        if suggestion:
                            action_result = handle_agentic_suggestion(suggestion)
                            if action_result and not action_result.startswith("[AGENT] No matching tool"):
                                jarvis_history.append(f"[Agent Action]: {action_result}")
                                assist.TTS(action_result)
                                # Do NOT speak the action result itself
                            # If no matching tool, do not speak


                    jarvis_history.append(f"Jarvis: {response}")
                    assist.TTS(response)

                    if response is None:
                        response = ""
                    skip_hot_word_check = "?" in response
                    recorder.start()
                    live_transcript = "" # Clear after processing

        # Draw Audio Feed
        audio_feed_content = [
            "Live Transcript:",
            live_transcript
        ]
        for i, line in enumerate(audio_feed_content):
            line_surface = SMALL_FONT.render(line, True, WHITE)
            screen.blit(line_surface, (audio_feed_rect.left + 10, audio_feed_rect.top + 50 + i * 30))

        # Draw Jarvis Transcriptions
        for i, entry in enumerate(jarvis_history):
            text_surface = SMALL_FONT.render(entry, True, WHITE)
            screen.blit(text_surface, (jarvis_transcription_rect.left + 10, jarvis_transcription_rect.top + 30 + i * 30))

        # Draw borders for the sections
        pygame.draw.rect(screen, GRAY, scene_capture_rect, 2)
        pygame.draw.rect(screen, GRAY, audio_feed_rect, 2)
        pygame.draw.rect(screen, GRAY, jarvis_transcription_rect, 2)

        # Draw section titles
        draw_section_title(screen, "Scene Capture", scene_capture_rect)
        draw_section_title(screen, "Audio Feed", audio_feed_rect)
        draw_section_title(screen, "Jarvis Transcriptions", jarvis_transcription_rect)


        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    recorder.stop()
    camera_manager.release()
    sys.exit()

if __name__ == '__main__':
    run() 