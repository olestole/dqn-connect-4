from connect_x import ConnectX
from agent import Agent
from dqn import DQN
import os
import numpy as np

CHECKPOINT_DIR = "../checkpoints"
HISTORY_DIR = "../history"
WEIGHT_ITERATION = 10

def main():
    env = ConnectX()

    # Create the training agent with prelaoded weights
    target_network_weights_path = os.path.join(CHECKPOINT_DIR, f"main_{WEIGHT_ITERATION}_1900")
    target_network = DQN(env.observation_space.shape, env.action_space.n)
    target_network.load_weights(target_network_weights_path)
    main_network = DQN(env.observation_space.shape, env.action_space.n)
    main_network_weights_path = os.path.join(CHECKPOINT_DIR, f"main_{WEIGHT_ITERATION}_1900")
    main_network.load_weights(main_network_weights_path)
    training_agent = Agent(env, main_network, target_network)

    state = env.reset()
    done = env.is_done()
    while (True):
        # Agent turn
        action = training_agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()
        while (not info['legal_move'] and not done):
                action = training_agent.get_action(state)
                state, reward, done, info = env.step(action)
                env.render()
        
        if (done): break

        # Your turn
        print(f"Valid positions: {env.valid_positions()}")
        col = int(input("Enter column: "))
        state, reward, done, info = env.step(col)
        while (not info['legal_move'] and not done):
            env.render()
            print(f"Valid positions: {env.valid_positions()}")
            col = int(input("Enter column: "))
            state, reward, done, info = env.step(col)
        
        env.render()
        if (done): break
        
    
    print()
    print(env.board)

if __name__ == "__main__":
    main()
    
