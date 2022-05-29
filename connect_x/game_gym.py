from connect_x import ConnectX
import gym
from agent import Agent
from randomAgent import RandomAgent
import numpy as np

def calculate_reward(player, reward):
    if (player == 1):
        return reward
    else:
        return -reward


def main():
    env = ConnectX(start_player = 1)
    training_agent = Agent(env)
    opponent_agent = RandomAgent(env)
    rewards = []
    
    for _ in range(1000):
        observation = env.reset()
        print(f"Starting player: {env.start_player}")
        for t in range(1000):
            # env.render()
            player = training_agent if env.player == 1 else opponent_agent
            action = player.get_action(observation)

            # print(f"Timestep: {t} \t Action: {action}")
            observation, reward, done, info = env.step(action)
            
            while (not info['legal_move'] and not done):
                action = player.get_action(observation)
                # print(f"Timestep: {t} \t Action: {action}")
                observation, reward, done, info = env.step(action)
            if done:
                # print(observation, reward, done, info)
                reward = calculate_reward(info['player'], reward)
                rewards.append(reward)
                # env.render()
                print(f"\n-----\nEpisode finished after {t+1} timesteps:\nreward: {reward}\ndone: {done}\ninfo: {info}\n-----\n")
                break
    env.close()

    print(rewards)
    rewards = np.array(rewards)
    print(np.count_nonzero(rewards == 1), np.count_nonzero(rewards == -1))

if __name__ == "__main__":
    main()