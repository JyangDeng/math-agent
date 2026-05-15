#study_planner.py
from llm.client import client


def generate_study_plan(weak_points, days=7):

    prompt = f"""
学生薄弱知识点：

{weak_points}

请生成：
1. {days}天学习计划
2. 每天学习内容
3. 推荐刷题量
4. 学习建议
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