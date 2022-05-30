import os
from connect_x import ConnectX
from agent import Agent
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
    "HISTORY_N": 11,
    "CHECKPOINT_N": 10,
    "WEIGHT_ITERATION": 9,
    "N_EPISODES": 2000,
    "N_WARMUP_EPISODES": 100,
    "TEST": False
}

def main():
    history = History()
    env = ConnectX(start_player = 1)
    
    # Create the training agent with prelaoded weights
    target_network_weights_path = os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['WEIGHT_ITERATION']}_900")
    target_network = DQN(env.observation_space.shape, env.action_space.n)
    target_network.load_weights(target_network_weights_path)
    main_network = DQN(env.observation_space.shape, env.action_space.n)
    main_network_weights_path = os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['WEIGHT_ITERATION']}_900")
    main_network.load_weights(main_network_weights_path)
    training_agent = Agent(env, main_network, target_network)
    
    # Create the opponent agent
    opponent_network_weights_path = os.path.join(run_settings['CHECKPOINT_DIR'], f"main_{run_settings['WEIGHT_ITERATION']}_0")
    opponent_network = DQN(env.observation_space.shape, env.action_space.n)
    opponent_network.load_weights(opponent_network_weights_path)
    opponent_agent = Agent(env, opponent_network, opponent_network)
    
    memory = ReplayBuffer()

    if (run_settings['TEST']):
        run_dqn(run_settings['N_EPISODES'], env, main_network, target_network, memory, training_agent, opponent_agent, \
            history, run_settings, training=False)
        return history

    # Warmup: Create samples for the replay buffer
    run_dqn(run_settings['N_WARMUP_EPISODES'], env, main_network, target_network, memory, training_agent, opponent_agent, \
        history, run_settings, training=False)

    # Use the replay buffer to train the agent
    run_dqn(run_settings['N_EPISODES'], env, main_network, target_network, memory, training_agent, opponent_agent, \
        history, run_settings, training=True)

    return history

if __name__ == "__main__":
    history = None
    history_file_path = os.path.join(run_settings['HISTORY_DIR'], f"{run_settings['HISTORY_N']}_history.pkl")

    try:
        history = main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    pickle.dump(history, open(history_file_path, "wb"))