#router.py
def route_question(question: str):

    q = question.lower()

    # 学习计划
    if "学习计划" in q or "复习计划" in q:
        return "study_plan"

    # 练习生成
    if "练习" in q or "出题" in q:
        return "exercise"

    # 答案检查
    if "我的答案" in q or "我选" in q:
        return "check_answer"

    # 默认知识问答
    return "qa"