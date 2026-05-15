#wrong_book.py

import json

WRONG_PATH = "memory/wrong_questions.json"


def add_wrong_question(question, user_answer, correct_answer):

    try:
        with open(WRONG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = []

    data.append({
        "question": question,
        "user_answer": user_answer,
        "correct_answer": correct_answer
    })

    with open(WRONG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)