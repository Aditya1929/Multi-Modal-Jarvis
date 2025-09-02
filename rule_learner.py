import json
import os
from collections import Counter
from itertools import combinations

LOGS_DIR = 'multimodal_logs'
EVENT_LOG_FILE = os.path.join(LOGS_DIR, 'event_log.json')
LEARNED_RULES_FILE = os.path.join(LOGS_DIR, 'learned_rules.json')

# Parameters
MIN_SUPPORT = 5  # Minimum number of times a pattern must occur

# Load event log
def load_event_log():
    if not os.path.exists(EVENT_LOG_FILE):
        print(f"No event log found at {EVENT_LOG_FILE}")
        return []
    with open(EVENT_LOG_FILE, 'r') as f:
        return json.load(f)

def extract_keywords(transcript):
    # Simple keyword extraction: split on spaces, lowercase, remove punctuation
    if not transcript:
        return set()
    words = transcript.lower().replace('.', '').replace(',', '').split()
    return set(words)

def main():
    events = load_event_log()
    if not events:
        print("No events to process.")
        return

    # Collect (frozenset(objects), frozenset(keywords)) pairs for positive feedback
    pattern_counter = Counter()
    for event in events:
        if event['event_type'] == 'proactive_action' and event.get('feedback') == 'positive':
            for ctx in event['context']:
                objects = set(ctx.get('objects', []))
                keywords = extract_keywords(ctx.get('transcript', ''))
                if objects and keywords:
                    # Consider all pairs of objects and keywords
                    for obj in objects:
                        for kw in keywords:
                            pattern_counter[(frozenset([obj]), frozenset([kw]))] += 1
                    # Also consider the full set
                    pattern_counter[(frozenset(objects), frozenset(keywords))] += 1

    # Find frequent patterns
    candidate_rules = []
    for (obj_set, kw_set), count in pattern_counter.items():
        if count >= MIN_SUPPORT:
            rule = {
                "objects": list(obj_set),
                "keywords": list(kw_set),
                "action": "<TO_BE_DEFINED>",
                "status": "pending",
                "support": count
            }
            candidate_rules.append(rule)

    # Save to learned_rules.json
    with open(LEARNED_RULES_FILE, 'w') as f:
        json.dump(candidate_rules, f, indent=4)
    print(f"Wrote {len(candidate_rules)} candidate rules to {LEARNED_RULES_FILE}")

if __name__ == '__main__':
    main() 