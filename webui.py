# web/webui.py

import streamlit as st
import json

from agent.math_agent import MathAgent

from learning.profile import analyze_weak_points
from learning.exercise_generator import generate_exercise
from learning.study_planner import generate_study_plan

# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="考研数学智能学习Agent",
    page_icon="📘",
    layout="wide"
)

# =========================
# 初始化 Agent
# =========================

agent = MathAgent()

# =========================
# 页面标题
# =========================

st.title("📘 考研数学智能学习 Agent")

st.caption(
    "RAG + ReAct + Tool Calling + SymPy + Learning Profile"
)

# =========================
# 侧边栏
# =========================

st.sidebar.title("📚 学习功能")

menu = st.sidebar.radio(
    "请选择功能",
    [
        "智能问答",
        "学习画像",
        "错题本",
        "生成练习",
        "学习计划"
    ]
)

# =========================
# 清空聊天记录
# =========================

if st.sidebar.button("🗑 清空聊天记录"):

    st.session_state.messages = []

    st.sidebar.success("聊天记录已清空")

# =========================
# 初始化聊天记录
# =========================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ==========================================================
# 1. 智能问答
# ==========================================================

if menu == "智能问答":

    st.header("💬 智能问答")

    # =========================
    # 显示历史消息
    # =========================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # =========================
    # 用户输入
    # =========================

    prompt = st.chat_input("请输入数学问题...")

    # =========================
    # 处理输入
    # =========================

    if prompt:

        # 保存用户消息

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # 显示用户消息

        with st.chat_message("user"):

            st.markdown(prompt)

        # Assistant 回复

        with st.chat_message("assistant"):

            with st.spinner("Agent正在思考中..."):

                result = agent.solve(prompt)

                answer = result["final_answer"]

                # Markdown + LaTeX 渲染

                st.markdown(answer)

                # =========================
                # Debug 推理过程
                # =========================

                if "steps" in result:

                    with st.expander("🧠 Agent推理过程"):

                        for step in result["steps"]:

                            st.markdown(
                                f"### Step {step['step']}"
                            )

                            st.markdown(
                                f"**Thought:** {step['thought']}"
                            )

                            st.markdown(
                                f"**Tool:** `{step['tool']}`"
                            )

                            st.markdown(
                                f"**Args:** `{step['args']}`"
                            )

                            st.markdown(
                                f"**Result:** `{step['result']}`"
                            )

                            st.divider()

        # 保存 Assistant 回复

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

# ==========================================================
# 2. 学习画像
# ==========================================================

elif menu == "学习画像":

    st.header("📊 学习画像")

    weak_points = analyze_weak_points()

    if not weak_points:

        st.info("暂无学习记录")

    else:

        st.subheader("近期高频知识点")

        st.bar_chart(weak_points)

        st.markdown("---")

        for k, v in weak_points.items():

            st.markdown(f"- **{k}** ：{v} 次")

# ==========================================================
# 3. 错题本
# ==========================================================

elif menu == "错题本":

    st.header("❌ 错题本")

    try:

        with open(
            "memory/wrong_questions.json",
            "r",
            encoding="utf-8"
        ) as f:

            wrong_questions = json.load(f)

    except:

        wrong_questions = []

    if len(wrong_questions) == 0:

        st.info("暂无错题")

    else:

        for idx, item in enumerate(wrong_questions):

            with st.expander(f"错题 {idx + 1}"):

                st.markdown("### 📌 题目")

                st.markdown(item["question"])

                st.markdown("### ❌ 你的答案")

                st.markdown(item["user_answer"])

                st.markdown("### ✅ 正确答案")

                st.markdown(item["correct_answer"])

# ==========================================================
# 4. 生成练习
# ==========================================================

elif menu == "生成练习":

    st.header("📝 AI练习生成")

    topic = st.text_input(
        "请输入知识点",
        placeholder="例如：矩阵特征值"
    )

    difficulty = st.selectbox(
        "请选择难度",
        [
            "基础",
            "中等",
            "考研真题风格"
        ]
    )

    if st.button("生成练习题"):

        if topic.strip() == "":

            st.warning("请输入知识点")

        else:

            with st.spinner("正在生成练习题..."):

                exercise = generate_exercise(
                    f"{topic}，难度：{difficulty}"
                )

            st.success("生成完成")

            st.markdown(exercise)

# ==========================================================
# 5. 学习计划
# ==========================================================

elif menu == "学习计划":

    st.header("📅 个性化学习计划")

    weak_points = analyze_weak_points()

    if not weak_points:

        st.info("暂无学习记录")

    else:

        st.subheader("当前学习情况")

        for k, v in weak_points.items():

            st.markdown(f"- {k}：{v} 次")

        st.markdown("---")

        study_days = st.slider(
            "请选择学习周期（天）",
            3,
            30,
            7
        )

        if st.button("生成学习计划"):

            with st.spinner("正在生成学习计划..."):

                plan = generate_study_plan({
                    "weak_points": weak_points,
                    "days": study_days
                })

            st.success("学习计划生成完成")

            st.markdown(plan)