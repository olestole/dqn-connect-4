import os
from connect_x import ConnectX
from agent import Agent
from policy_window import PolicyWindow
from dqn import DQN
from replay_buffer import ReplayBuffer
import logging
from history import History
import pickle
from game_logic import run_dqn

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)



run_settings = {
    "CHECKPOINT_DIR": "../checkpoints",
    "HISTORY_DIR": "../history",
    "CHECKPOINT_ITERATIONS": 100,
    "NETWORK_SYNC_ITERATIONS": 300,
    "HISTORY_N": 18,
    "CHECKPOINT_N": 15,
    "WEIGHT_ITERATION": 14,
    "N_EPISODES": 610,
    "N_WARMUP_EPISODES": 100,
    "OPPONENT_LAG": 3,
    "TEST": False
}

def main():
    history_file_path = os.path.join(run_settings['HISTORY_DIR'], f"{run_settings['HISTORY_N']}_history.pkl")
    history = History()
    env = ConnectX(start_player = 1)
    
    # Create the training agent with prelaoded weights
    target_network_weights_path = os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['WEIGHT_ITERATION']}_9600")
    target_network = DQN(env.observation_space.shape, env.action_space.n)
    target_network.load_weights(target_network_weights_path)
    main_network = DQN(env.observation_space.shape, env.action_space.n)
    main_network_weights_path = os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['WEIGHT_ITERATION']}_9600")
    main_network.load_weights(main_network_weights_path)
    training_agent = Agent(env, main_network, target_network)
    
    # Create the opponent agent
    opponent_network_weights_path = os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['WEIGHT_ITERATION']}_00")
    opponent_network = DQN(env.observation_space.shape, env.action_space.n)
    opponent_network.load_weights(opponent_network_weights_path)
    opponent_agent = Agent(env, opponent_network, opponent_network)
    
    memory = ReplayBuffer()
    policy_window = PolicyWindow(size=5)

    if (run_settings['TEST']):
        run_dqn(run_settings['N_EPISODES'], env, main_network, target_network, memory, training_agent, opponent_agent, \
            history, run_settings, 0, training=False)
        with open(history_file_path, "wb") as f:
            logging.info("Backing up the history")
            pickle.dump(history, f)
        return

    # Warmup: Create samples for the replay buffer
    run_dqn(run_settings['N_WARMUP_EPISODES'], env, main_network, target_network, memory, training_agent, opponent_agent, \
        history, run_settings, 0, training=False)

    logging.info("\nFinished warmup\n")

    for epoch in range(200):
        logging.info(f"Starting new run\t{epoch}")
        run_dqn(run_settings['N_EPISODES'], env, main_network, target_network, memory, training_agent, opponent_agent, \
            history, run_settings, epoch, training=True)
        policy_window.add(main_network.get_weights())

        if (epoch % run_settings['OPPONENT_LAG'] == 0):
                # Change the opponent's policy into a lag of the current training_agent main_network
                new_policy_weights = policy_window.get()
                logging.info("Setting the new policy")
                opponent_network.set_weights(new_policy_weights)
        
        with open(history_file_path, "wb") as f:
            logging.info("Backing up the history")
            pickle.dump(history, f)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")