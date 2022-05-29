from connect_x import ConnectX
import math
from agent import Agent
from dqn import DQN
from replay_buffer import ReplayBuffer
from randomAgent import RandomAgent
import numpy as np

def calculate_reward(player, reward):
    if (player == 1):
        return reward
    else:
        return -reward

def exploration_rate(n: int, min_rate=0.01) -> float:
        """Decaying exploration rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((n  + 1) / 25)))

def learning_rate(n: int, min_rate=0.01) -> float:
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def one_step(env, state, action, memory):
    next_state, reward, done, info = env.step(action)
    memory.add(state, action, reward, next_state, done)
    env.render()
    return next_state, reward, done, info

def main():
    env = ConnectX(start_player = 1)
    main_network = DQN(env.observation_space.shape, env.action_space.n)
    target_network = DQN(env.observation_space.shape, env.action_space.n)
    training_agent = Agent(env, main_network, target_network)
    opponent_agent = RandomAgent(env)
    memory = ReplayBuffer()
    rewards = []
    
    for episode in range(10):
        epsilon = exploration_rate(episode)
        lr = learning_rate(episode)

        observation = env.reset()
        for t in range(1000):
            # Start player takes action
            action = training_agent.get_action_epsilon_greedy(epsilon, observation)
            observation, reward, done, info = one_step(env, observation, action, memory)

            while (not info['legal_move'] and not done):
                action = training_agent.get_action_epsilon_greedy(epsilon, observation)
                observation, reward, done, info = one_step(env, observation, action, memory)

            if (done):
                rewards.append(calculate_reward(1, reward))
                print(f"\n-----\nEpisode finished after {t+1} timesteps:\nreward: {reward}\ndone: {done}\ninfo: {info}\n-----\n")
                break

            # Opponent takes action
            action = opponent_agent.get_action(observation)
            observation, reward, done, info = one_step(env, observation, action, memory)

            while (not info['legal_move'] and not done):
                action = opponent_agent.get_action(observation)
                observation, reward, done, info = one_step(env, observation, action, memory)

            if (done):
                rewards.append(calculate_reward(2, reward))
                print(f"\n-----\nEpisode finished after {t+1} timesteps:\nreward: {reward}\ndone: {done}\ninfo: {info}\n-----\n")
                break

    env.close()

    rewards = np.array(rewards)
    print(np.count_nonzero(rewards == 1), np.count_nonzero(rewards == -1))

if __name__ == "__main__":
    main()