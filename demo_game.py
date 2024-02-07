from Assets import checkers_model
from Agents.random_agent import RandomAgent
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.models import model_from_json
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import json

def load_model(board_json_path, model_weights_path):
    """
    Loads a model from a JSON file and a set of weights.

    Parameters:
    - board_json: str, the path to the JSON file containing the model architecture.
    - model_weights_path: str, the path to the file containing the model weights.

    Returns:
    - model: keras.Model, the loaded model.
    """
    with open(board_json_path, 'r') as json_file:
        board_json = json_file.read()

    reinforced_model = model_from_json(board_json)
    reinforced_model.load_weights(model_weights_path)
    reinforced_model.compile(optimizer='adadelta', loss='mean_squared_error')
    return reinforced_model

def best_move(boards, reinforced_model):
    scores = reinforced_model.predict_on_batch(boards)
    max_index = np.argmax(scores)
    return max_index

def simulate_game(reinforced_model, custom_agent):
    game = checkers_model.CheckersGame(8)
    game.render()
    while game.game_winner() == 0:
        possible_actions = game.possible_actions(1)
        if possible_actions is None:
            break
        action_index = best_move(game.simulate_next_boards_player_1(), reinforced_model)
        action = game.possible_actions(1)[action_index]
        
        state, _ = game.step(action, 1)
        print(f"Player 1")
        print(game.render())
        print("-----------------------------------------------------------")
        # ------------------------------------------------
        action = custom_agent.select_action(game)
        
        if len(game.possible_actions(-1)) == 0:
            break
        state, _ = game.step(action, custom_agent.player_id)
        print(f"Player -1")
        print(game.render())
    return game.game_winner()  

def main():
    reinforced_model = load_model('trained_models/morning/reinforced_model.json', 'trained_models/morning/reinforced_model.h5')
    custom_agent = RandomAgent(-1)
    winner = simulate_game(reinforced_model, custom_agent)
    print(f"Winner: {winner}")
main()