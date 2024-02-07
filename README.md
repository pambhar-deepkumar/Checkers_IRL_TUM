# Checkers Game

This project is a Checkers game that uses Q-Learning for training the AI.

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

Configuration

Before running the project, configure the paths in the setup.py file to specify where the models and other outputs should be stored. This includes paths for saving the Q table and the evaluation results after training. After setting up you should run setup.py.



You can then edit the parameters in train_and_evaluate_ql.py for Q-Learning:
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

Please replace `<name_of_venv>` with the name you want to give to your virtual environment.
