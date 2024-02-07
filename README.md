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

### How to run 
Please ucomment the code as required in the model_training.py. And run following command

```bash
python model_training.py
```

> Note : Please configure the paths in the yaml file for storeing the models.
