import numpy as np
from connect_x import ConnectX
from dqn import DQN

class Agent():
    def __init__(self, env: ConnectX, main_network: DQN, target_network: DQN):
        self.env = env
        self.main_network = main_network
        self.target_network = target_network
    
    def get_action(self, observation):
        q_values = self.main_network.predict(observation.reshape((1, 6, 7)))
        action = np.argmax(q_values)
        return action

    def get_action_epsilon_greedy(self, epsilon, observation):
        if (np.random.random() < epsilon):
            return self.env.action_space.sample()
        else:
            return self.get_action(observation)