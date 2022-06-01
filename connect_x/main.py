import os
from connect_x import ConnectX
from agent import Agent
from trainer import Trainer
from policy_window import PolicyWindow
from dqn import DQN
from replay_buffer import ReplayBuffer
import logging
from history import History

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

run_settings = {
    "CHECKPOINT_DIR": "../checkpoints",
    "HISTORY_DIR": "../history",
    "RENDER": False,
    "CHECKPOINT_ITERATIONS": 300,
    "NETWORK_SYNC_ITERATIONS": 300,
    "HISTORY_N": 35,
    "CHECKPOINT_N": 26,
    "WEIGHT_ITERATION": 25,
    "N_EPISODES": 1000,
    "N_WARMUP_EPISODES": 200,
    "OPPONENT_LAG": 2,
}

def main():
    start_player = 1
    env = ConnectX(start_player = 1) # BUG: start_player should always be 1, so that the logic doesn't counteract eachother
    history = History()
    memory = ReplayBuffer()
    policy_window = PolicyWindow(size=5)
    
    # Create the training agent with prelaoded weights
    target_network_weights_path = os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['WEIGHT_ITERATION']}_171200")
    target_network = DQN(env.observation_space.shape, env.action_space.n, initial_weights_path=target_network_weights_path, network_model=3)
    main_network_weights_path = os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['WEIGHT_ITERATION']}_171200")
    main_network = DQN(env.observation_space.shape, env.action_space.n, initial_weights_path=main_network_weights_path, network_model=3)
    training_agent = Agent(env, main_network, target_network)
    
    # Create the opponent agent
    opponent_network_weights_path = os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['WEIGHT_ITERATION']}_00")
    opponent_network = DQN(env.observation_space.shape, env.action_space.n, initial_weights_path=None, network_model=3, name=opponent_network_weights_path)
    opponent_agent = Agent(env, opponent_network, opponent_network)

    trainer = Trainer(run_settings, env, history, memory, policy_window, training_agent, opponent_agent, start_player)
    
    # trainer.training_loop(change_turns=True, change_epochs=3)
    trainer.test_loop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warn("Interrupted by user")
