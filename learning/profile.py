#profile.py
import json
from collections import Counter

HISTORY_PATH = "memory/history.json"


def save_learning_record(question, knowledge_points):

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    except:
        history = []

    history.append({
        "question": question,
        "knowledge_points": knowledge_points
    })

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def analyze_weak_points():

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    except:
        return {}

    counter = Counter()

    for item in history:

        for kp in item["knowledge_points"]:
            counter[kp] += 1

    return dict(counter.most_common(10))