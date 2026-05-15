from llm.client import client


def generate_exercise(topic):

    prompt = f"""
请围绕以下知识点生成一道考研数学练习题：

知识点:
{topic}

要求：
1. 给出题目
2. 给出答案
3. 给出详细解析
"""

    response = client.chat.completions.create(
        model="qwen3.6-max-preview",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    return response.choices[0].message.content