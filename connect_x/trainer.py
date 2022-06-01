import os
import math
import numpy as np
import logging

def generate_ckpt_path(run_settings, episode, epoch):
    return os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['CHECKPOINT_N']}_{epoch}{episode}")

def generate_history_path(run_settings):
    return os.path.join(run_settings['HISTORY_DIR'], f"{run_settings['HISTORY_N']}_history.pkl")

def calculate_epsilon(n: int, min_rate=0.05) -> float:
    """Decaying exploration rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n  + 1) / 70)))

class Trainer():
    def __init__(self, run_settings, env, history, memory, policy_window, training_agent, opponent_agent, start_player):
        self.run_settings = run_settings
        self.env = env
        self.history = history
        self.memory = memory
        self.policy_window = policy_window
        self.training_agent = training_agent
        self.opponent_agent = opponent_agent
        self.epsilon = 1.0
        self.opponent_epsilon = 1.0
        self.epoch = 0
        self.episode = 0
        self.episode_move = 0
        self.history_path = generate_history_path(self.run_settings)
        self.start_player = start_player
    
    def change_start_player(self):
        if (self.start_player == 1):
            self.start_player = 2
        else:
            self.start_player = 1
    
    def test_loop(self):
        self.episodes_loop(self.run_settings['N_EPISODES'], warmup=False, testing=True)
        self.history.save(self.history_path)
    
    def training_loop(self, change_turns=False, change_epochs=3):
        self.episodes_loop(self.run_settings['N_WARMUP_EPISODES'], warmup=True)
        for epoch in range(100):
            logging.info(f"Starting new run\t{epoch}")
            self.epoch = epoch
            self.episodes_loop(self.run_settings['N_EPISODES'], warmup=False)
            self.policy_window.add(self.training_agent.main_network.get_weights())
            self.handle_opponent_policy_change()
            self.history.save(self.history_path)
            
            if (change_turns and epoch % change_epochs == 1):
                self.change_start_player()
                logging.info(f"Changed start player, new start player is: {self.start_player}")
    
    def player_step(self, state, epsilon, agent, training_agent=False, warmup=False, testing=False):
        # Agent takes action
        initial_state = np.copy(state)
        action = agent.get_action_epsilon_greedy(epsilon, state)
        state, reward, done, info = self.one_step(action)

        while (not info['legal_move'] and not done):
            if (training_agent):
                self.memory.add(self.mask_board(initial_state), action, reward, self.mask_board(np.copy(state)), done)
            action = agent.get_action_epsilon_greedy(epsilon, state)
            state, reward, done, info = self.one_step(action)

        is_done = self.handle_if_done(done, info, state, action, reward, initial_state, training_agent=training_agent, training=not (testing or warmup))
        return state, is_done, reward, info, action


    def episodes_loop(self, n_episodes, warmup=False, testing=False):
        for episode in range(n_episodes):
            self.episode = episode
            self.epsilon = calculate_epsilon(episode) if not testing else 0 # If testing, always be greedy 
            self.opponent_epsilon = calculate_epsilon(episode) if not testing else 1 # If testing, always be random  
            state = self.env.reset()
            initial_state = np.copy(state)
            for episode_move in range(1000):
                self.episode_move = episode_move
                
                # Start player takes action
                state, done, a1_reward, info, a1_action = self.player_step(state, self.epsilon if self.start_player == 1 else self.opponent_epsilon, self.training_agent if self.start_player == 1 else self.opponent_agent, training_agent=self.start_player == 1, warmup=warmup, testing=testing)
                if (done): break

                if (self.start_player == 2):
                    initial_state = np.copy(state)

                # Opponent takes action
                state, done, a2_reward, info, a2_action = self.player_step(state, self.opponent_epsilon if self.start_player == 1 else self.epsilon, self.opponent_agent if self.start_player == 1 else self.training_agent, training_agent=self.start_player == 2, warmup=warmup, testing=testing)
                if (done): break
                
                if (self.start_player == 2):
                    self.memory.add(self.mask_board(initial_state), a2_action, a2_reward, self.mask_board(np.copy(state)), done)

                if (self.start_player == 1):
                    self.memory.add(self.mask_board(initial_state), a1_action, a1_reward, self.mask_board(np.copy(state)), done)
                    initial_state = np.copy(state)

            if (episode % 25 == 0):
                mean_reward = np.array(self.history.get_last_n("player_1_reward", 50)).mean()
                logging.info(f"Episode: {episode} / {self.run_settings['N_EPISODES']}\tMean Reward last 50 episodes: {mean_reward}")

    def one_step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward = self.calculate_reward(info, done) # The reward should always be in the perspective of agent_1
        if (self.run_settings['RENDER']): self.env.render()
        return next_state, reward, done, info
    
    def handle_if_done(self, done, info, state, action, reward, initial_state, training_agent=True, training=True):
        if (not done):
            return False
        
        reward = self.calculate_reward(info, done)
        self.memory.add(self.mask_board(initial_state), action, reward, self.mask_board(np.copy(state)), done)
        loss = self.training_step() if training else None
        self.history.add_tuples([("player_1_reward", reward), ("winner", info['winner']), ("done_type", info['done_type']), ("final_state", state), ("game_length", self.episode_move), ("opponent_policy", self.opponent_agent.main_network.name), ("loss", loss), ("start_player", self.start_player)])
        return True

    def handle_opponent_policy_change(self):
        # Change the opponent's policy into a lag of the current training_agent main_network
        if (self.epoch % self.run_settings['OPPONENT_LAG'] == 1):
                new_policy_weights = self.policy_window.get()
                logging.info("Setting the new policy")
                self.opponent_agent.change_network_weights(new_policy_weights)
    
    def handle_weights_ckpt(self, ckpt_path):
        if (self.episode % self.run_settings['CHECKPOINT_ITERATIONS'] == 0):
            logging.info(f"Saving checkpoint, {self.episode}")
            self.training_agent.main_network.save_weights(ckpt_path)
    
    def handle_network_sync(self, ckpt_path):
        if (self.episode % self.run_settings['NETWORK_SYNC_ITERATIONS'] == 0):
            logging.info(f"Syncing target_network with previous {self.run_settings['NETWORK_SYNC_ITERATIONS']} episode main_network")
            self.training_agent.target_network.load_weights(ckpt_path)


    def training_step(self, batch_size=32):
        ckpt_path = generate_ckpt_path(self.run_settings, self.episode, self.epoch)
        mini_batch = self.memory.sample(batch_size)
        loss = self.training_agent.main_network.update(mini_batch, 0.95, self.training_agent.target_network)
        
        self.handle_weights_ckpt(ckpt_path)
        self.handle_network_sync(ckpt_path)
        return loss
    
    def mask_board(self, state):
        # Mask the board to show only the player's pieces
        masked_board = np.copy(state)
        if (self.start_player == 1):
            masked_board[masked_board == 2] = -1
        else:
            masked_board[masked_board == 1] = -1
            masked_board[masked_board == 2] = 1
        return masked_board
    
    def calculate_reward(self, info, done):
        """
        Win:
        - Vertical              + 0.5
        - Horizontal            + 3
        - Diagonal              + 3

        Loss:
        - Vertical              - 1
        - Horizontal            - 1
        - Diagonal              - 1

        No winner:
        - Illegal move:         - 10
        - Enemy illegal move:   + 0
        - Draw:                 + 0
        """
        if (not done):
            return 0
        
        # No winner
        if (info['winner'] == 0):
            # Training_agent played 2+ illegal moves
            if (info['done_type'] == 'illegal_move' and info['player'] == 1):
                return -10
            elif (info['done_type'] == 'illegal_move' and info['player'] == 2):
                return 0
            else:
                return 0

        if (info['winner'] == self.start_player):
            # Training_agent won
            if (info['done_type'] == 'vertical'):
                return 0.5
            elif (info['done_type'] == 'horizontal'):
                return 3
            elif (info['done_type'] == 'diagonal'):
                return 3
        
        if (info['winner'] != self.start_player):
            # Training_agent lost
            if (info['done_type'] == 'vertical'):
                return -1
            elif (info['done_type'] == 'horizontal'):
                return -1
            elif (info['done_type'] == 'diagonal'):
                return -1