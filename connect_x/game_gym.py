import os
from connect_x import ConnectX
import math
from agent import Agent
from dqn import DQN
from replay_buffer import ReplayBuffer
from randomAgent import RandomAgent
import numpy as np
import logging

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.WARN)

CHECKPOINT_DIR = "../checkpoints/connect_x_dqn_checkpoint.pth"
CHECKPOINT_ITERATIONS = 100
NETWORK_SYNC_ITERATIONS = 1000

def calculate_reward(player, reward):
    if (player == 1):
        return reward
    else:
        return -reward

def exploration_rate(n: int, min_rate=0.01) -> float:
        """Decaying exploration rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((n  + 1) / 25)))

def one_step(env, state, action, memory):
    next_state, reward, done, info = env.step(action)
    memory.add(state, action, reward, next_state, done)
    env.render()
    return next_state, reward, done, info

def training_step(main_network: DQN, target_network: DQN, memory, episode, batch_size=32):
    # Sample a random minibatch of N transitions from replay buffer
    mini_batch = memory.sample(batch_size)
    main_network.update(mini_batch, 0.95, target_network)
    if (episode % CHECKPOINT_ITERATIONS == 0):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"main_{episode}")
        main_network.save_weights(checkpoint_path)
    if (episode % NETWORK_SYNC_ITERATIONS == 0):
        target_network.load_weights(checkpoint_path)

def run_dqn(n_episodes, env, main_network, target_network, memory, training_agent, opponent_agent, rewards, training = True):
    for episode in range(n_episodes):
        epsilon = exploration_rate(episode)

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
                logging.info(f"\n-----\nEpisode finished after {t+1} timesteps:\nreward: {reward}\ndone: {done}\ninfo: {info}\n-----\n")
                if (training):
                    training_step(main_network, target_network, memory, episode, batch_size = 32)
                break

            # Opponent takes action
            action = opponent_agent.get_action(observation)
            observation, reward, done, info = one_step(env, observation, action, memory)

            while (not info['legal_move'] and not done):
                action = opponent_agent.get_action(observation)
                observation, reward, done, info = one_step(env, observation, action, memory)

            if (done):
                rewards.append(calculate_reward(2, reward))
                logging.info(f"\n-----\nEpisode finished after {t+1} timesteps:\nreward: {reward}\ndone: {done}\ninfo: {info}\n-----\n")
                break

    env.close()


def main():
    env = ConnectX(start_player = 1)
    main_network = DQN(env.observation_space.shape, env.action_space.n)
    target_network = DQN(env.observation_space.shape, env.action_space.n)
    training_agent = Agent(env, main_network, target_network)
    opponent_agent = RandomAgent(env)
    memory = ReplayBuffer()
    rewards = []
    
    # Warmup: Create samples for the replay buffer
    run_dqn(100, env, main_network, target_network, memory, training_agent, opponent_agent, rewards, training=False)

    # Use the replay buffer to train the agent
    run_dqn(2000, env, main_network, target_network, memory, training_agent, opponent_agent, rewards, training=True)

    rewards = np.array(rewards)
    # print(np.count_nonzero(rewards == 1), np.count_nonzero(rewards == -1))

if __name__ == "__main__":
    main()