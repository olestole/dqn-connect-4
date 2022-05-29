from collections import deque
from this import s
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size = 10000):
        self.memory = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        mini_batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = [np.array([experience[key] for experience in mini_batch]) for key in range(5)]
        return states, actions, rewards, next_states, dones

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)