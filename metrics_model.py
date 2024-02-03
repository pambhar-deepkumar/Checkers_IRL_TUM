import checker_env
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Metrics model, which only looks at heuristic scoring metrics used for labeling
metrics_model = Sequential()
metrics_model.add(Dense(32, activation='relu', input_dim=9)) 
metrics_model.add(Dense(16, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))

# output is passed to relu() because labels are binary
metrics_model.add(Dense(1, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))
metrics_model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=["acc"])

start_board = checker_env.expand(checker_env.np_board())
boards_list = checker_env.generate_next(start_board)
branching_position = 0
nmbr_generated_game = 100
print("Generating game states...")
while len(boards_list) < nmbr_generated_game:
	temp = len(boards_list) - 1
	for i in range(branching_position, len(boards_list)):
		if (checker_env.possible_moves(checker_env.reverse(checker_env.expand(boards_list[i]))) > 0):
				boards_list = np.vstack((boards_list, checker_env.generate_next(checker_env.reverse(checker_env.expand(boards_list[i])))))
	branching_position = temp

# calculate/save heuristic metrics for each game state
metrics	= np.zeros((0, 9))
winning = np.zeros((0, 1))



print("Calculating metrics...")
for board in tqdm(boards_list[:nmbr_generated_game], desc="Calculating metrics"):
	temp = checker_env.get_metrics(board)
	metrics = np.vstack((metrics, temp[1:]))
	winning  = np.vstack((winning, temp[0]))
 

count_1 = np.count_nonzero(winning == 1)

count_min_1 = np.count_nonzero(winning == -1)
 
print(f"count_1: {count_1}, count_min_1: {count_min_1}")




print("Fitting metrics model...")
# fit the metrics model
history = metrics_model.fit(metrics , winning, epochs=32, batch_size=64, verbose=0)


def plot_history(history):
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

    # Print the parameters of the history object
    print("History Parameters: ", history.params)

# Call the function with the history object
# plot_history(history)

# =======================================================================================
# Board model
board_model = Sequential()

# input dimensions is 32 board position values
board_model.add(Dense(64 , activation='relu', input_dim=32))

# use regularizers, to prevent fitting noisy labels
board_model.add(Dense(32 , activation='relu', kernel_regularizer=regularizers.l2(0.01)))
board_model.add(Dense(16 , activation='relu', kernel_regularizer=regularizers.l2(0.01))) # 16
board_model.add(Dense(8 , activation='relu', kernel_regularizer=regularizers.l2(0.01))) # 8

# output isn't squashed, because it might lose information
board_model.add(Dense(1 , activation='linear', kernel_regularizer=regularizers.l2(0.01)))
board_model.compile(optimizer='nadam', loss='binary_crossentropy')

# calculate heuristic metric for data
metrics = np.zeros((0, 9))
winning = np.zeros((0, 1))
data = boards_list

for board in  tqdm(data, desc="Calculating metrics for data"):
	temp = checker_env.get_metrics(board)
	metrics = np.vstack((metrics, temp[1:]))
	winning  = np.vstack((winning, temp[0]))
 
# calculate probilistic (noisy) labels
probabilistic = metrics_model.predict_on_batch(metrics)
# fit labels to {-1, 1}
probabilistic = np.sign(probabilistic)

# calculate confidence score for each probabilistic label using error between probabilistic and weak label

confidence = 1/(1 + np.absolute(winning - probabilistic))

# pass to the Board model
board_model.fit(data, probabilistic, epochs=32, batch_size=64, sample_weight=confidence, verbose=0)

board_json = board_model.to_json()
with open('board_model.json', 'w') as json_file:
	json_file.write(board_json)
board_model.save_weights('board_model.h5')

print('Checkers Board Model saved to: board_model.json/h5')






json_file = open('board_model.json', 'r')
board_json = json_file.read()
json_file.close()

reinforced_model = model_from_json(board_json)
reinforced_model.load_weights('board_model.h5')
reinforced_model.compile(optimizer='adadelta', loss='mean_squared_error')

data = np.zeros((1, 32))
labels = np.zeros(1)
win = lose = draw = 0
winrates = []
learning_rate = 0.5
discount_factor = 0.95

for gen in tqdm(range(0, 10), desc="Processing Generations"):
	for game in range(0, 5):
		temp_data = np.zeros((1, 32))
		board = checker_env.expand(checker_env.np_board())
		player = np.sign(np.random.random() - 0.5)
		turn = 0
		while (True):
			moved = False
			boards = np.zeros((0, 32))
			if (player == 1):
				boards = checker_env.generate_next(board)
			else:
				boards = checker_env.generate_next(checker_env.reverse(board))

			scores = reinforced_model.predict_on_batch(boards)
			max_index = np.argmax(scores)
			best = boards[max_index]

			if (player == 1):
				board = checker_env.expand(best)
				temp_data = np.vstack((temp_data, checker_env.compress(board)))
			else:
				board = checker_env.reverse(checker_env.expand(best))

			player = -player

			# punish losing games, reward winners  & drawish games reaching more than 200 turns
			winner = checker_env.game_winner(board)
			if (winner == 1 or (winner == 0 and turn >= 200) ):
				if winner == 1:
					win = win + 1
				else:
					draw = draw + 1
				reward = 10
				old_prediction = reinforced_model.predict_on_batch(temp_data[1:])
				optimal_futur_value = np.ones(old_prediction.shape)
				temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_futur_value - old_prediction )
				data = np.vstack((data, temp_data[1:]))
				labels = np.vstack((labels, temp_labels))
				break
			elif (winner == -1):
				lose = lose + 1
				reward = -10
				old_prediction = reinforced_model.predict_on_batch(temp_data[1:])
				optimal_futur_value = -1*np.ones(old_prediction.shape)
				temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_futur_value - old_prediction )
				data = np.vstack((data, temp_data[1:]))
				labels = np.vstack((labels, temp_labels))
				break
			turn = turn + 1

		if ((game+1) % 200 == 0):
			reinforced_model.fit(data[1:], labels[1:], epochs=16, batch_size=256, verbose=0)
			data = np.zeros((1, 32))
			labels = np.zeros(1)
	winrate = int((win+draw)/(win+draw+lose)*100)
	winrates.append(winrate)
 
	reinforced_model.save_weights('reinforced_model.h5')
 
print('Checker_env Board Model updated by reinforcement learning & saved to: reinforced_model.json/h5')