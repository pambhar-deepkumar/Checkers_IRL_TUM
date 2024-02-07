# Checkers Game

This branch of the Checkers_IRL_TUM project is dedicated to training a Deep Q-Learning Agent for the Checkers game. The goal is to develop an agent that can learn effective strategies for playing Checkers through reinforcement learning techniques, specifically utilizing Deep Q-Learning.

## Setup

### Creating a Virtual Environment

Isolating the project dependencies is crucial for managing the project's environment. You can create a virtual environment using the following commands:
```python
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate  # On Windows
```
After activating your virtual environment, install the required dependencies:

- tensorflow-cpu
- numpy
- keras
- matplotlib
- tqdm
- pyyaml

## Configuration
Before running the project, configure the paths in the config.yaml file to specify where the models and other outputs should be stored. This includes paths for saving the metrics model, board model, and the reinforced model after training.


### How to run 
To start training the Deep Q-Learning Agent, you first need to uncomment the necessary sections in the model_training.py file as per your training phase requirement. Then, execute the following command:
```bash
python model_training.py
```

> Note : This script will initiate the training process, which includes generating game states, training the metrics model, and optionally the board model and reinforcement learning model based on the uncommented sections.

## Code Structure

The codebase is structured into several key components:

- checker_env: A custom module that simulates the Checkers game environment, including board initialization, generating possible moves, and evaluating game states.
- model_training.py: The main script for training the Deep Q-Learning Agent, which includes functions for training models, evaluating them, and saving the trained models.
- config.yaml: A configuration file used for specifying paths for saving models and other training parameters.

## Key Functions
- generate_game_states: Generates a set of game states to be used for training.
- train_metrics_model: Trains a model to predict winning moves based on game metrics.
- train_board_model: Trains a model based on the board state to predict the likelihood of winning.
- reinforce_model: Utilizes reinforcement learning to refine the strategies of the trained model.



