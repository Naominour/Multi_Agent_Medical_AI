class BaseAgent:
    def __init__(self, name):
        self.name = name
        self.state = {}

    def process(self, input_data):
        raise NotImplementedError("Each agent must implement its own process method.")

    def communicate(self, message):
        print(f"{self.name} received: {message}")
