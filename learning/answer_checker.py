#answer_checker.py

from llm.client import client


def check_answer(question, user_answer, reference_answer):

    prompt = f"""
题目：
{question}

学生答案：
{user_answer}

参考答案：
{reference_answer}

请完成：
1. 判断是否正确
2. 指出错误
3. 给出改进建议
"""

    response = client.chat.completions.create(
        model="qwen3.6-max-preview",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content