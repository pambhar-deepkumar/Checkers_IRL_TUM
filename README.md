# Checkers Game

This project is a Checkers game that uses Q-Learning for training the AI.

## Setup

### Creating a Virtual Environment

To isolate the dependencies of this project, it's recommended to use a virtual environment. You can create one using the following command:

```powershell
python -m venv <name_of_venv>
```
Installing Dependencies
After activating the virtual environment, you can install the necessary dependencies with:
```
pip install -r requirements.txt
```
Usage
Before running the scripts, you need to set up the folder structure by running setup.py.

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