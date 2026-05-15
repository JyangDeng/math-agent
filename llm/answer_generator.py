from llm.client import client


def generate_answer(
    question,
    knowledge,
    thought,
    tool,
    tool_result,
    reasoning_steps
):

    knowledge_text = "\n".join(knowledge)

    prompt = f"""
你是一名考研数学老师。

请根据：

1. 用户问题
2. 检索到的知识
3. Agent推理过程
4. 工具计算结果

生成最终数学回答。

要求：

- 使用中文
- 数学表达准确
- 条理清晰
- 如果是概念题，重点解释概念
- 如果是计算题，分步骤解答

------------------------

用户问题：
{question}

------------------------

相关知识：
{knowledge_text}

------------------------

Agent推理：
{thought}

------------------------

工具调用：
{tool}

------------------------

工具结果：
{tool_result}
"""

    response = client.chat.completions.create(

        model="qwen3.6-max-preview",

        messages=[
            {
                "role": "system",
                "content": "你是一名专业的考研数学教师。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3
    )

    return response.choices[0].message.content