DEBUG = False


def format_answer(agent_output):

    # 用户最终看到的答案
    final_answer = agent_output.get(
        "final_answer",
        "未生成答案"
    )

    # 非调试模式
    if not DEBUG:

        return final_answer

    # 调试模式
    sections = []

    sections.append("=== 最终答案 ===\n")

    sections.append(final_answer)

    sections.append("\n\n=== Agent推理 ===\n")

    sections.append(
        str(agent_output.get("thought"))
    )

    sections.append("\n\n=== 工具调用 ===\n")

    tool = agent_output.get("tool")

    if tool:
        sections.append(str(tool))
    else:
        sections.append("未调用工具")

    sections.append("\n\n=== 工具结果 ===\n")

    sections.append(
        str(agent_output.get("result"))
    )

    return "\n".join(sections)