import os
import math
from dqn import DQN
import numpy as np
import logging


def calculate_reward(info, done):
    """
    Win:
    - Vertical              + 0.5
    - Horizontal            + 2
    - Diagonal              + 2

    Loss:
    - Vertical              - 1
    - Horizontal            - 1
    - Diagonal              - 1

    No winner:
    - Illegal move:         - 5
    - Enemy illegal move:   + 0
    - Draw:                 + 0
    """
    if (not done):
        return 0
    
    # No winner
    if (info['winner'] == 0):
        # Training_agent played 2+ illegal moves
        if (info['done_type'] == 'illegal_move' and info['player'] == 1):
            return -5
        elif (info['done_type'] == 'illegal_move' and info['player'] == 2):
            return 0
        else:
            return 0

    if (info['winner'] == 1):
        # Training_agent won
        if (info['done_type'] == 'vertical'):
            return 0.5
        elif (info['done_type'] == 'horizontal'):
            return 2
        elif (info['done_type'] == 'diagonal'):
            return 2
    
    if (info['winner'] == 2):
        # Training_agent lost
        if (info['done_type'] == 'vertical'):
            return -1
        elif (info['done_type'] == 'horizontal'):
            return -1
        elif (info['done_type'] == 'diagonal'):
            return -1


def calculate_epsilon(n: int, min_rate=0.01) -> float:
        """Decaying exploration rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((n  + 1) / 25)))


def one_step(env, action):
    next_state, reward, done, info = env.step(action)
    reward = calculate_reward(info, done) # The reward should always be in the perspective of agent_1
    # env.render()
    return next_state, reward, done, info


def generate_ckpt_path(run_settings, episode, epoch):
    return os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['CHECKPOINT_N']}_{epoch}{episode}")


def training_step(main_network: DQN, target_network: DQN, memory, episode: int, run_settings, epoch, batch_size=32):

    checkpoint_path = generate_ckpt_path(run_settings, episode, epoch)
    mini_batch = memory.sample(batch_size)
    loss = main_network.update(mini_batch, 0.95, target_network)
    
    if (episode % run_settings['CHECKPOINT_ITERATIONS'] == 0):
        logging.info(f"Saving checkpoint, {episode}")
        main_network.save_weights(checkpoint_path)
    
    if (episode % run_settings['NETWORK_SYNC_ITERATIONS'] == 0):
        logging.info(f"Syncing target_network with previous 100 episode main_network, {episode - 100}")
        target_network.load_weights(checkpoint_path)
    return loss


def log_episode(episode: int, n_episodes):
    if (episode % 25 == 0):
        logging.info(f"Episode: {episode} / {n_episodes}")


def handle_if_done(done, info, memory, state, action, reward, initial_state, history, training, main_network, target_network, episode, move_n, run_settings, epoch, player = 1):
    if (not done):
        return False
    
    #  If player 1 fininshes the game before player 2 has taken its turn, the game should be added to memory
    if (player == 1):
        reward = calculate_reward(info, done)
        memory.add(initial_state, action, reward, np.copy(state), done)

    if (training):
        loss = training_step(main_network, target_network, memory, episode, run_settings, epoch, batch_size = 32)
        history.add("loss", loss)

    logging.info(f"Reward: {reward}")
    history.add_tuples([("player_1_reward", reward), ("winner", info['winner']), ("done_type", info['done_type']), ("final_state", state), ("game_length", episode)])
    logging.debug(f"\n-----\nEpisode finished after {move_n+1} timesteps:\nreward: {reward}\ndone: {done}\ninfo: {info}\n-----\n")
    return True


def run_dqn(n_episodes, env, main_network, target_network, memory, training_agent, opponent_agent, history, run_settings, epoch, training = True):
    for episode in range(n_episodes):
        log_episode(episode, run_settings['N_EPISODES'])
        epsilon = calculate_epsilon(episode)
        opponent_epsilon = 0 if training else 1 # Random if testing, greedy when training
        state = env.reset()
        
        for move_n in range(1000):
            # Copy the initial state so that we have a copy when adding it to the replay buffer
            initial_state = np.copy(state)
            
            # Start player takes action
            action = training_agent.get_action_epsilon_greedy(epsilon, state)
            state, reward, done, info = one_step(env, action)

            while (not info['legal_move'] and not done):
                memory.add(initial_state, action, reward, np.copy(state), done)
                action = training_agent.get_action_epsilon_greedy(epsilon, state)
                state, reward, done, info = one_step(env, action)

            is_done = handle_if_done(done, info, memory, state, action, reward, initial_state, history, training, main_network, target_network, episode, move_n, run_settings, epoch)
            if (is_done): break

            # Opponent takes action
            opponent_action = opponent_agent.get_action_epsilon_greedy(opponent_epsilon, state) # Greedy action
            state, reward, done, info = one_step(env, opponent_action)

            while (not info['legal_move'] and not done):
                opponent_action = opponent_agent.get_action_epsilon_greedy(opponent_epsilon, state) # Greedy action
                state, reward, done, info = one_step(env, opponent_action)

            # Add to replay buffer since both agents have made a move
            memory.add(initial_state, action, reward, np.copy(state), done)

            is_done = handle_if_done(done, info, memory, state, action, reward, initial_state, history, training, main_network, target_network, episode, move_n, run_settings, epoch, player = 2)
            if (is_done): break