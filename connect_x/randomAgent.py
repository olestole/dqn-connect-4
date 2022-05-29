import math
from connect_x import ConnectX

class RandomAgent():
    def __init__(self, env: ConnectX):
        self.env = env
    
    def get_action(self, observation):
        return self.env.action_space.sample()

    