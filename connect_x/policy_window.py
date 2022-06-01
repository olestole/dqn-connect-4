from collections import deque
import numpy as np

class PolicyWindow():
    def __init__(self, size: int = 5) -> None:
        self.window = deque(maxlen=size)
    
    def add(self, model) -> None:
        self.window.append(model)
    
    def get(self):
        return self.window.pop()
    
    def sample(self):
        # BUG: There's an issue with this implementation
        i = np.random.randint(0, len(self.window))
        policy = self.window[i]
        self.window.remove(policy)
        return policy