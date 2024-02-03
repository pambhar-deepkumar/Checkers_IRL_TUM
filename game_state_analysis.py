import checker_env
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

# Get the current date and time
def get_current_time_string():
    """Utility function to get the current date and time in a file-friendly format."""
    now = datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')

def train_metrics_model(metrics, winning, epochs=32, batch_size=64, input_dim=9, regularization=0.1, save_path=None):
    """
    Trains a metrics model using the given metrics and winning data.

    Parameters:
    - metrics (numpy.ndarray): The input metrics data.
    - winning (numpy.ndarray): The target winning data.
    - epochs (int): The number of epochs to train the model (default: 32).
    - batch_size (int): The batch size for training (default: 64).
    - input_dim (int): The input dimension of the model (default: 9).
    - regularization (float): The regularization strength (default: 0.1).
    - save_path (str): The path to save the trained model (default: None).

    Returns:
    - metrics_model (keras.models.Sequential): The trained metrics model.
    - history (keras.callbacks.History): The training history.
    """
    if save_path is None:
        date_time_string = get_current_time_string()
        model_filename = f'metrics_model_{date_time_string}.h5'
        json_filename = f'metrics_model_{date_time_string}.json'
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_filename = os.path.join(save_path, 'metrics_model.h5')
        json_filename = os.path.join(save_path, 'metrics_model.json')
    
    metrics_model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(regularization)),
        Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(regularization))  # Using sigmoid for binary classification
    ])
    metrics_model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=["acc"])
    print("Fitting metrics model...")
    history = metrics_model.fit(metrics, winning, epochs=epochs, batch_size=batch_size, verbose=0)
    metrics_model.save_weights(model_filename)
    with open(json_filename, 'w') as json_file:
        json_file.write(metrics_model.to_json())
    print(f'Checkers Metrics Model saved to: {json_filename} and {model_filename}')
    
    return metrics_model, history

def generate_game_states(minimum_nmbr_generated_game):
    """
    Generates game states for a given number of games.

    Args:
        nmbr_generated_game (int): The number of game states to generate.

    Returns:
        tuple: A tuple containing the generated game states, metrics, and winning information.
            - boards_list (numpy.ndarray): An array of game boards representing the generated game states.
            - metrics (numpy.ndarray): An array of heuristic metrics calculated for each game state.
            - winning (numpy.ndarray): An array indicating whether each game state is a winning state (1) or not (0).
    """
    start_board = checker_env.expand(checker_env.np_board())
    boards_list = checker_env.generate_next(start_board)
    branching_position = 0
    print("Generating game states...")
    while len(boards_list) < minimum_nmbr_generated_game:
        temp = len(boards_list) - 1
        for i in range(branching_position, len(boards_list)):
            if (checker_env.possible_moves(checker_env.reverse(checker_env.expand(boards_list[i]))) > 0):
                    boards_list = np.vstack((boards_list, checker_env.generate_next(checker_env.reverse(checker_env.expand(boards_list[i])))))
        branching_position = temp

    # calculate/save heuristic metrics for each game state
    metrics	= np.zeros((0, 9))
    winning = np.zeros((0, 1))

    print("Calculating metrics...")
    for board in tqdm(boards_list, desc="Calculating metrics"):
        temp = checker_env.get_metrics(board)
        metrics = np.vstack((metrics, temp[1:]))
        winning  = np.vstack((winning, temp[0]))
        
    return boards_list, metrics, winning
    
def plot_history_metrics_model(history):
    """
    This function plots the model's accuracy and loss history.
    """
    # List of metrics in the history
    metrics = list(history.history.keys())
    
    # Check if 'acc' and 'loss' are in the metrics list
    if 'acc' in metrics and 'loss' in metrics:
        # Create subplots for accuracy and loss
        fig, axs = plt.subplots(2)
        
        # Plotting Accuracy
        axs[0].plot(history.history['acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['Train', 'Validation'], loc='upper left')
        
        # Plotting Loss
        axs[1].plot(history.history['loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['Train', 'Validation'], loc='upper left')
        
        # Display the plots
        plt.tight_layout()
        plt.show()
        
    else:
        print("The history object does not contain 'acc' or 'loss'.")


def train_board_model(data, metrics, winning, metrics_model, epochs=32, batch_size=64, save_path=None):
    
    if save_path is None:
        date_time_string = get_current_time_string()
        model_filename = f'board_model_{date_time_string}.h5'
        json_filename = f'board_model_{date_time_string}.json'
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_filename = os.path.join(save_path, 'board_model.h5')
        json_filename = os.path.join(save_path, 'board_model.json')
    
    board_model = Sequential()

    # input dimensions is 32 board position values
    board_model = Sequential([
        Dense(64, activation='relu', input_dim=32),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.01))
    ])
    board_model.compile(optimizer='nadam', loss='binary_crossentropy')
    probabilistic = metrics_model.predict_on_batch(metrics)
    probabilistic = np.sign(probabilistic)

    # calculate confidence score for each probabilistic label using error between probabilistic and weak label

    confidence = 1/(1 + np.absolute(winning - probabilistic))

    # pass to the Board model
    board_model.fit(data, probabilistic, sample_weight=confidence, epochs=epochs, batch_size=batch_size, verbose=0)
    
    board_model.save_weights(model_filename)
    with open(json_filename, 'w') as json_file:
        json_file.write(board_model.to_json())

    print(f'Checkers Board Model saved to: {json_filename} and {model_filename}')

    return board_model



def reinforce_model(board_model = None, model_json_path = None, model_weights_path = None, 
                               num_generations=500, games_per_generation=200, 
                               learning_rate=0.5, discount_factor=0.95, save_path=None ):
    # Load the model
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_save_path = os.path.join(save_path, 'reinforced_model.h5')
        params_save_path = os.path.join(save_path, 'reinforcement_params.json')
        model_json_filename = os.path.join(save_path, 'reinforced_model.json')
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_save_path = f'reinforced_model_{timestamp}.h5'
        params_save_path = f'reinforcement_params_{timestamp}.json'
        model_json_filename = f'reinforced_model_{timestamp}.json'

    if not board_model:    
        with open(model_json_path, 'r') as json_file:
            board_json = json_file.read()
        reinforced_model = model_from_json(board_json)
        reinforced_model.load_weights(model_weights_path)
        reinforced_model.compile(optimizer='adadelta', loss='mean_squared_error')

    else:
        reinforced_model = board_model
    # Initialize variables
    data = np.zeros((1, 32))  # Assuming the board representation is 32 units
    labels = np.zeros(1)
    win = lose = draw = 0
    winrates = []

    # Run generations
    for gen in tqdm(range(num_generations), desc="Generations"):
        for game in range(games_per_generation):
            temp_data = np.zeros((1, 32))
            board = checker_env.expand(checker_env.np_board())
            player = np.sign(np.random.random() - 0.5)
            turn = 0
            while True:
                if player == 1:
                    boards = checker_env.generate_next(board)
                else:
                    boards = checker_env.generate_next(checker_env.reverse(board))

                scores = reinforced_model.predict_on_batch(boards)
                max_index = np.argmax(scores)
                best_board = boards[max_index]

                if player == 1:
                    board = checker_env.expand(best_board)
                    temp_data = np.vstack((temp_data, checker_env.compress(board)))
                else:
                    board = checker_env.reverse(checker_env.expand(best_board))
                player = -player

                winner = checker_env.game_winner(board)
                if winner in [1, -1] or (winner == 0 and turn >= 200):
                    win += winner == 1
                    draw += winner == 0
                    lose += winner == -1
                    reward = 10 if winner == 1 else -10
                    old_prediction = reinforced_model.predict_on_batch(temp_data[1:])
                    optimal_future_value = np.ones(old_prediction.shape) if winner == 1 else -1 * np.ones(old_prediction.shape)
                    temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_future_value - old_prediction)
                    data = np.vstack((data, temp_data[1:]))
                    labels = np.vstack((labels, temp_labels))
                    break
                turn += 1

        if gen % 10 == 0 or gen == num_generations - 1:  # Optionally adjust the frequency
            reinforced_model.fit(data[1:], labels[1:], epochs=16, batch_size=256, verbose=0)
            data = np.zeros((1, 32))
            labels = np.zeros(1)
        
        # Calculate and store winrate for the current generation
        winrate = int((win + draw) / (win + draw + lose) * 100)
        winrates.append(winrate)
        # Reset counters
        win = lose = draw = 0

    model_json = reinforced_model.to_json()
    with open(model_json_filename, 'w') as json_file:
        json_file.write(model_json)

    reinforced_model.save_weights(model_save_path)
    print(f'Reinforced Model and architecture saved to: {model_save_path} and {model_json_filename}')


    # Save the parameters used for reinforcement
    params = {
        'num_generations': num_generations,
        'games_per_generation': games_per_generation,
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'model_save_path': model_save_path,
        'winrates': winrates    
    }
    with open(params_save_path, 'w') as f:
        json.dump(params, f, indent=4)

    print(f'Reinforcement parameters saved to: {params_save_path}')

    return winrates

