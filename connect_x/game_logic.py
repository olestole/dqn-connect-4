import os
import math
from dqn import DQN
import numpy as np
import logging

def calculate_reward(player, reward, illegal_moves = 0):
    if (player == 1):
        return reward
    # Don't punish agent1 when agent2 makes illegal_moves
    if (illegal_moves >= 2):
        return 0
    return -reward

def exploration_rate(n: int, min_rate=0.01) -> float:
        """Decaying exploration rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((n  + 1) / 25)))

def one_step(env, state, action, memory, add_to_memory=True):
    next_state, reward, done, info = env.step(action)
    reward = calculate_reward(info['player'], reward, illegal_moves=info['illegal_moves']) # The reward should always be in the perspective of agent_1
    env.render()
    return next_state, reward, done, info

def training_step(main_network: DQN, target_network: DQN, memory, episode: int, ckpt_dir, ckpt_n, ckpt_iter, network_sync_iter, batch_size=32):
    checkpoint_path = os.path.join(ckpt_dir, f"main_{ckpt_n}_{episode}")
    # Sample a random minibatch of N transitions from replay buffer
    mini_batch = memory.sample(batch_size)
    loss = main_network.update(mini_batch, 0.95, target_network)
    if (episode % ckpt_iter == 0):
        logging.info(f"Saving checkpoint, {episode}")
        main_network.save_weights(checkpoint_path)
    if (episode % network_sync_iter == 0):
        logging.info(f"Syncing target_network with previous 100 episode main_network, {episode - 100}")
        target_network.load_weights(checkpoint_path)
    return loss

def log_episode(episode: int):
    if (episode % 25 == 0):
        logging.info(f"Episode: {episode}")

def run_dqn(n_episodes, env, main_network, target_network, memory, training_agent, opponent_agent, history, ckpt_dir, ckpt_n, ckpt_iter, network_sync_iter, training = True):
    for episode in range(n_episodes):
        log_episode(episode)
        epsilon = exploration_rate(episode)
        state = env.reset()
        for t in range(1000):
            initial_state = np.copy(state)
            # Start player takes action
            action = training_agent.get_action_epsilon_greedy(epsilon, state)
            state, reward, done, info = one_step(env, state, action, memory)

            while (not info['legal_move'] and not done):
                memory.add(initial_state, action, reward, np.copy(state), done) # Add to memory if agent places illegal move
                action = training_agent.get_action_epsilon_greedy(epsilon, state)
                state, reward, done, info = one_step(env, state, action, memory)

            if (done):
                reward = calculate_reward(1, reward, illegal_moves=info['illegal_moves'])
                history.add("player_1_reward", reward)
                history.add("winner", info['winner'])
                history.add("done_type", info['done_type'])
                history.add("final_state", state)
                history.add("game_length", t)
                if (training):
                    loss = training_step(main_network, target_network, memory, episode, ckpt_dir, ckpt_n, ckpt_iter, network_sync_iter,  batch_size = 32 )
                    history.add("loss", loss)
                logging.debug(f"\n-----\nEpisode finished after {t+1} timesteps:\nreward: {reward}\ndone: {done}\ninfo: {info}\n-----\n")
                break

            opponent_action = opponent_agent.get_action_epsilon_greedy(0, state) # Opponent takes action - always random, due to epsilon = 0
            state, reward, done, info = one_step(env, state, opponent_action, memory, add_to_memory=False)

            while (not info['legal_move'] and not done):
                opponent_action = opponent_agent.get_action_epsilon_greedy(0, state)
                state, reward, done, info = one_step(env, state, opponent_action, memory, add_to_memory=False)

            memory.add(initial_state, action, reward, np.copy(state), done)

            if (done):
                history.add("player_1_reward", reward)
                history.add("winner", info['winner'])
                history.add("done_type", info['done_type'])
                history.add("final_state", state)
                history.add("game_length", t)
                if (training):
                    loss = training_step(main_network, target_network, memory, episode, ckpt_dir, ckpt_n, ckpt_iter, network_sync_iter, batch_size = 32)
                    history.add("loss", loss)
                logging.debug(f"\n-----\nEpisode finished after {t+1} timesteps:\nreward: {reward}\ndone: {done}\ninfo: {info}\n-----\n")
                break

    env.close()