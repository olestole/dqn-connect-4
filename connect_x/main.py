import os
from connect_x import ConnectX
from agent import Agent
from dqn import DQN
from replay_buffer import ReplayBuffer
from randomAgent import RandomAgent
import logging
from history import History
import pickle
from game_logic import run_dqn

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CHECKPOINT_DIR = "../checkpoints"
HISTORY_DIR = "../history"
CHECKPOINT_ITERATIONS = 100
NETWORK_SYNC_ITERATIONS = 300
HISTORY_N = 9
CHECKPOINT_N = 9
WEIGHT_ITERATION = CHECKPOINT_N - 1
N_EPISODES = 500
N_WARMUP_EPISODES = 0
TEST = True

def main():
    history = History()
    env = ConnectX(start_player = 1)
    
    # Create the training agent with prelaoded weights
    target_network_weights_path = os.path.join(CHECKPOINT_DIR, f"main_{WEIGHT_ITERATION}_3000")
    target_network = DQN(env.observation_space.shape, env.action_space.n)
    target_network.load_weights(target_network_weights_path)
    main_network = DQN(env.observation_space.shape, env.action_space.n)
    main_network_weights_path = os.path.join(CHECKPOINT_DIR, f"main_{WEIGHT_ITERATION}_3000")
    main_network.load_weights(main_network_weights_path)
    training_agent = Agent(env, main_network, target_network)
    
    # Create the opponent agent
    opponent_network_weights_path = os.path.join(CHECKPOINT_DIR, f"main_{WEIGHT_ITERATION}_0")
    opponent_network = DQN(env.observation_space.shape, env.action_space.n)
    opponent_network.load_weights(opponent_network_weights_path)
    opponent_agent = Agent(env, opponent_network, opponent_network)
    
    memory = ReplayBuffer()

    if (TEST):
        run_dqn(N_EPISODES, env, main_network, target_network, memory, training_agent, opponent_agent, \
            history, CHECKPOINT_DIR, CHECKPOINT_N, CHECKPOINT_ITERATIONS, NETWORK_SYNC_ITERATIONS, training=False)
        return history

    # Warmup: Create samples for the replay buffer
    run_dqn(N_WARMUP_EPISODES, env, main_network, target_network, memory, training_agent, opponent_agent, \
        history, CHECKPOINT_DIR, CHECKPOINT_N, CHECKPOINT_ITERATIONS, NETWORK_SYNC_ITERATIONS, training=False)

    # Use the replay buffer to train the agent
    run_dqn(N_EPISODES, env, main_network, target_network, memory, training_agent, opponent_agent, \
        history, CHECKPOINT_DIR, CHECKPOINT_N, CHECKPOINT_ITERATIONS, NETWORK_SYNC_ITERATIONS, training=True)

    return history

if __name__ == "__main__":
    history = None
    history_file_path = os.path.join(HISTORY_DIR, f"{HISTORY_N}_history.pkl")

    try:
        history = main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    pickle.dump(history, open(history_file_path, "wb"))