from collections import deque
from this import s
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size = 2000):
        self.memory = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        mini_batch = [self.memory[i] for i in indices]
        return mini_batch

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)