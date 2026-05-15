 

from agent.math_agent import MathAgent
from output.formatter import format_answer

def main():
    agent = MathAgent()
    
    while True:
        question = input("\n请输入数学题(输入exit退出): ")
        if question == "exit":
            break
        
        result = agent.solve(question)
        print("\n" + format_answer(result))

if __name__ == "__main__":
    main()