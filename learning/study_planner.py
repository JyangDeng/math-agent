#study_planner.py

from llm.client import client


def generate_study_plan(weak_points):

    prompt = f"""
学生当前薄弱知识点：

{weak_points}

请生成：

1. 3天复习计划
2. 每天学习内容
3. 推荐练习方向
4. 时间安排
"""

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5
    )

    return response.choices[0].message.content