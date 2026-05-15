#answer_checker.py

from llm.client import client


def check_answer(question, correct_answer, student_answer):

    prompt = f"""
题目：
{question}

标准答案：
{correct_answer}

学生答案：
{student_answer}

请：
1. 判断对错
2. 分析错误原因
3. 给出学习建议
"""

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content