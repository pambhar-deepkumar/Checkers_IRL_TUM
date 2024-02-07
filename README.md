# Checkers Game
This branch of the Checkers_IRL_TUM project aims to create a competitive AI for playing Checkers using Q-Learning, a reinforcement learning algorithm. It features a modular design allowing for easy experimentation with different AI strategies and opponent models.

## Project Overview

Project Structure
- Agents/: Contains different agent implementations including Q-Learning Agent, Random Agent, and AlphaBeta Agent.
- Assets/: Includes the Checkers game model and utilities for managing game state and rules.
- trained_models/: Directory for storing trained models and their weights.
- setup.py: Configuration script for setting up project paths and environment variables.
- requirements.txt: Lists all the project dependencies for easy installation.
- train_and_evaluate_ql.py: Main script for training the Q-Learning agent and evaluating its performance against other agents.
- README.md: Provides project documentation, setup instructions, and usage guidelines.

## Setup

### Creating a Virtual Environment


Isolating the project dependencies is crucial for managing the project's environment. You can create a virtual environment using the following commands:
```python
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate  # On Windows
```
After activating the virtual environment, you can install the necessary dependencies with:
```
pip install -r requirements.txt
```

## Configuration

Before running the project, configure the paths in the setup.py file to specify where the models and other outputs should be stored. This includes paths for saving the Q table and the evaluation results after training. After setting up you should run setup.py.


### Adjust training parameters in train_and_evaluate_ql.py for Q-Learning:

```Python
# Training parameters
NUM_EPISODES = 10
EVALUATION_INTERVAL = 5
EVALUATION_GAMES = 1
FINAL_EVALUATION_GAMES = 10
GAMEBOARD_SIZE = 8
```

After setting the parameters, you can start the training and evaluation process with:
```
python train_and_evaluate_ql.py
```

## Output
The output you should get after executing train_and_evaluation_ql.py: 

### All the Q Value tables in a folder: 
![alt text](https://github.com/pambhar-deepkumar/Checkers_IRL_TUM/blob/qlearning/Assets/QTablejson.jpg)

### The Q Value table in the JSON format:
![alt text](https://github.com/pambhar-deepkumar/Checkers_IRL_TUM/blob/qlearning/Assets/Qlearning.jpg)


## Customization Guide

- Agent Strategies: Implement new strategies in the Agents/ directory. Use the Agent base class as a template.
- Game Rules: Modify game logic in Assets/checkers_model.py to experiment with different rule sets or board sizes.
- Training Parameters: Adjust parameters in train_and_evaluate_ql.py for different training durations, evaluation frequencies, and game settings.

