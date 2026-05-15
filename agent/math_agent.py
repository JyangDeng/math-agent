from rag.retriever import retrieve_knowledge
from learning.profile import save_learning_record
from llm.planner import plan
from llm.answer_generator import generate_answer

from tools.executor import execute_tool

from agent.state import AgentState

from agent.router import route_question

from agent.exercise_generator import generate_exercise

from agent.study_planner import generate_study_plan

from agent.answer_checker import check_answer

from agent.diagnosis import (
    get_weak_points,
    update_profile
)

from config import DEBUG, MAX_STEPS


class MathAgent:

    def __init__(self):

        self.state = AgentState()

        self.execution_cache = set()

        self.DEBUG = DEBUG

        self.MAX_STEPS = MAX_STEPS

    def solve(self, question):

        # =====================================
        # 0. 路由用户请求
        # =====================================

        mode = route_question(question)

        if self.DEBUG:
            print(f"\n[Router] 当前模式: {mode}")

        # =====================================
        # 学习计划模式
        # =====================================

        if mode == "study_plan":

            weak_points = get_weak_points()

            final_answer = generate_study_plan(
                weak_points=weak_points,
                days=7
            )

            return {
                "mode": mode,
                "final_answer": final_answer
            }

        # =====================================
        # 练习生成模式
        # =====================================

        if mode == "exercise":

            final_answer = generate_exercise(question)

            return {
                "mode": mode,
                "final_answer": final_answer
            }

        # =====================================
        # 答案评价模式
        # =====================================

        if mode == "check_answer":

            # 这里只是示例
            # 后面你可以接数据库/记忆系统

            final_answer = check_answer(
                question="示例题目",
                correct_answer="A",
                student_answer=question
            )

            return {
                "mode": mode,
                "final_answer": final_answer
            }

        # =====================================
        # 默认：知识问答模式
        # =====================================

        print("\n[1] 正在检索知识库...\n")

        knowledge = retrieve_knowledge(question)

        if self.DEBUG:
            for k in knowledge:
                print(k)

        print("-" * 50)

        # =====================================
        # 初始化 Observation
        # =====================================

        observations = []

        # =====================================
        # ReAct 推理循环
        # =====================================

        for step in range(self.MAX_STEPS):

            print(f"\n[Step {step + 1}] Agent正在思考...\n")

            # =====================================
            # 防止死循环
            # =====================================

            if step > 0:

                recent_actions = (
                    observations[-3:]
                    if len(observations) >= 3
                    else observations
                )

                if len(recent_actions) >= 3:

                    if all(
                        a["tool"] == recent_actions[0]["tool"]
                        and a["args"] == recent_actions[0]["args"]
                        for a in recent_actions
                    ):

                        print("[警告] 检测到重复操作，终止循环")

                        break

            # =====================================
            # Planner 推理
            # =====================================

            try:

                action = plan(
                    question=question,
                    retrieved_knowledge=knowledge,
                    history=observations
                )

            except Exception as e:

                return {
                    "final_answer": f"Planner调用失败: {str(e)}",
                    "steps": observations
                }

            if self.DEBUG:
                print(action)

            thought = action.get("thought", "")

            tool = action.get("tool")

            args = action.get("args", {})

            # =====================================
            # 无需工具
            # =====================================

            if tool is None:

                print("\n[Agent] 不再需要工具调用\n")

                break

            # =====================================
            # 防止重复执行
            # =====================================

            action_hash = f"{tool}_{str(args)}"

            if action_hash in self.execution_cache:

                print(f"[警告] 重复操作: {tool}")

                observations.append({
                    "step": step + 1,
                    "thought": thought,
                    "tool": tool,
                    "args": args,
                    "result": "跳过重复操作"
                })

                continue

            self.execution_cache.add(action_hash)

            # =====================================
            # 工具执行
            # =====================================

            print(f"\n[Step {step + 1}] 执行工具: {tool}\n")

            try:

                result = execute_tool(tool, args)

            except Exception as e:

                result = f"工具执行失败: {str(e)}"

            print("工具结果:")
            print(result)

            # =====================================
            # 保存 Observation
            # =====================================

            observation = {
                "step": step + 1,
                "thought": thought,
                "tool": tool,
                "args": args,
                "result": str(result)
            }

            observations.append(observation)

            self.state.add_step(
                thought=thought,
                tool=tool,
                result=result
            )

        # =====================================
        # 最终答案生成
        # =====================================

        print("\n[Final] 正在生成最终答案...\n")

        try:

            if observations:

                last_step = observations[-1]

                thought = last_step.get("thought", "")

                tool = last_step.get("tool")

                tool_result = str(
                    last_step.get("result", "")
                )

            else:

                thought = ""

                tool = None

                tool_result = ""

            final_answer = generate_answer(
                question=question,
                knowledge=knowledge,
                thought=thought,
                tool=tool,
                tool_result=tool_result,
                reasoning_steps=observations
            )

        except Exception as e:

            final_answer = f"答案生成失败: {str(e)}"

        # =====================================
        # 学情诊断（简单版）
        # =====================================

        try:

            if "极限" in question:
                update_profile("极限", correct=True)

            elif "积分" in question:
                update_profile("积分", correct=True)

            elif "级数" in question:
                update_profile("无穷级数", correct=True)

        except Exception as e:

            if self.DEBUG:
                print(f"[Diagnosis Error] {e}")

        save_learning_record(
            question,
            knowledge
        )

        # =====================================
        # 返回结果
        # =====================================

        return {
            "mode": "qa",
            "question": question,
            "knowledge": knowledge,
            "steps": observations,
            "final_answer": final_answer
        }