#diagnosis.py
import json

PROFILE_PATH = "memory/student_profile.json"


def load_profile():

    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)



def save_profile(profile):

    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=4)



def update_profile(topic, correct=False):

    profile = load_profile()

    if topic not in profile:
        profile[topic] = 0.5

    if correct:
        profile[topic] += 0.05
    else:
        profile[topic] -= 0.1

    profile[topic] = max(0, min(1, profile[topic]))

    save_profile(profile)



def get_weak_points(top_n=3):

    profile = load_profile()

    sorted_topics = sorted(profile.items(), key=lambda x: x[1])

    return sorted_topics[:top_n]