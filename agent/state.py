class AgentState:
    def __init__(self):
        self.history = []
        self.tool_results = []
    
    def add_step(self, thought, tool, result):
        self.history.append({
            "thought": thought,
            "tool": tool,
            "result": str(result)
        })