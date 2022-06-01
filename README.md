# DQN agent learning to excel in Connect 4 through self-play

## About The Project

Projectwork for [Autonomous and Adaptive Systems M](https://www.unibo.it/en/teaching/course-unit-catalogue/course-unit/2021/477337), University of Bologna 2022. This project aims to create a reinforcement learning agent capable of correctly playing and winning against random agents in the game of Connect 4. Connect 4 is a board game with a large state space which makes it infeasible to solve with brute force search. By implementing a deep reinforcement learning agent, taught by self-play, this project aims for good results against weak opponents. Although the agent is prone to overfit certain policies, after experimenting and implementing different agents, self-play methods and environment-tweaks, the agent learns the legal actions and is able to beat a random agent 83.7% of the time.

## Getting Started

### Prerequisites

- python 3.7.x
- pip

### Installation

- [Create a venv](#create-a-virtual-environment-venv) to install the python-packages into locally.

## Usage

The trained weights and run-history can be extracted from the zips `history/history.zip` and `checkpoints/weights.zip`.
Define the run settings in `main` by pointing it to the weights- and history-directory, and set the training/testing parameters.

After having configured the parameters in `main`, run the script with:
```sh
# /dqn-connect-4
$ python ./connect_x/main
```

## Misc

### Create a Virtual Environment (venv)

```bash
# /dqn-connect-4

$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

# If you want to install new packages
$ pip install <name-of-new-pip-package>

# When you're done
$ deactivate
```
