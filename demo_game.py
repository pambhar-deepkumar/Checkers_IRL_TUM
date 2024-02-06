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

def load_model(board_json, model_weights_path):
    """
    Loads a model from a JSON file and a set of weights.

    Parameters:
    - board_json: str, the path to the JSON file containing the model architecture.
    - model_weights_path: str, the path to the file containing the model weights.

    Returns:
    - model: keras.Model, the loaded model.
    """
    json_string = json.load(open('trained_models/reinforced_model.json'))
    print(json_string)
    reinforced_model = model_from_json(json_string)
    reinforced_model.load_weights(model_weights_path)
    reinforced_model.compile(optimizer='adadelta', loss='mean_squared_error')
    return reinforced_model

def best_move(boards, reinforced_model):
    scores = reinforced_model.predict_on_batch(boards)
    max_index = np.argmax(scores)
    return max_index

def simulate_game(reinforced_model, custom_agent):
    game = checkers_model.CheckersGame(8)
    while game.game_winner() == 0:
        possible_actions = game.possible_actions(1)
        if possible_actions is None:
            break
        action_index = best_move(game.simulate_next_boards_player_1, reinforced_model)
        action = game.possible_actions(1)[action_index]
        
        state, _ = game.step(action, 1)
        

        # ------------------------------------------------
        action = custom_agent.select_action(game)
        if game.possible_actions(custom_agent.player_id) is None:
            break
        state, _ = game.step(action, custom_agent.player_id)
        
    return game.game_winner()  

def main():
    reinforced_model = load_model('trained_models/reinforced_model.json', 'trained_models/reinforced_model.h5')
    custom_agent = RandomAgent(-1)
    winner = simulate_game(reinforced_model, custom_agent)
    if winner == 1:
        print('Reinforced model wins')
    elif winner == -1:
        print('Custom agent wins')
    else:
        print('It is a draw')

main()