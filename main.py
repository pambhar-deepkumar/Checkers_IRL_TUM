import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import checkers_env

import random

import matplotlib.pyplot as plt
import numpy as np
import CNN


def initialize_board():
    # 1 and -1 represent the pieces of two players 1 and -1
    board = np.zeros((6, 6))
    for i in range(2):
        for j in range(0, 6, 2):
            board[i][j + (i % 2)] = 1
            board[6 - i - 1][j + (i % 2)] = -1
    return board


env = checkers_env.CheckersGame(initialize_board(), 1)
env.render()
starters = env.possible_pieces(1)
env.possible_actions(player=1)
env.step([1, 1, 2, 0],1)
env.render()

# batch_size is the number of transitions sampled from the replay buffer
batch_size = 64
n_positions = 36

appro = CNN(n_positions)
appro.load_

def logistic(samples, targets):
    clf = LogisticRegression()
    clf.fit(samples, targets)
    return clf

