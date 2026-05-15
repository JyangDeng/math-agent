#exercise_generator.py

from llm.client import client


def generate_exercise(topic):

    prompt = f"""
请围绕知识点：

{topic}

生成：
1. 一道选择题或者解答题
2. 如果是选择题,生成四个选项
3. 正确答案
4. 详细解析

格式清晰。
"""

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    return response.choices[0].message.content