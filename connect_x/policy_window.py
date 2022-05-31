from collections import deque

class PolicyWindow():
    def __init__(self, size: int = 5) -> None:
        self.window = deque(maxlen=size)
    
    def add(self, model) -> None:
        self.window.append(model)
    
    def get(self):
        return self.window.pop()