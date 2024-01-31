import numpy as np
import random
from Assets.checkers_model import CheckersGame
from Agents.random_agent import RandomAgent
from Agents.alphabeta_agent import AlphaBetaAgent
from Agents.agent_base import Agent
import random
import json
import logging
import random


# Training parameters
NUM_EPISODES = 100
EVALUATION_INTERVAL = 50
EVALUATION_GAMES = 10
FINAL_EVALUATION_GAMES = 100

# Log file basenames
TRAINING_LOG_BASENAME = './ignored_files/game_training'
PERIODIC_EVALUATION_LOG_BASENAME = './ignored_files/periodic_evaluation'
EVALUATION_RESULTS_FILENAME = './ignored_files/evaluation_results.json'
FINAL_EVALUATION_LOG_BASENAME = './ignored_files/final_evaluation'

# Save file basenames
Q_TABLE_SAVE_BASENAME = './ignored_files/q_table'

# ==================================================================
def save_q_table(q_table, filename):
    # Convert the Q-table keys to strings
    q_table_str_keys = {str(key): value for key, value in q_table.items()}

    # Write to a file
    with open(filename, 'w') as file:
        json.dump(q_table_str_keys, file)
        
def load_q_table(filename):
    with open(filename, 'r') as file:
        q_table_str_keys = json.load(file)
    
    # Convert string keys back to tuple keys
    q_table = {eval(key): value for key, value in q_table_str_keys.items()}
    return q_table


# ==================================================================
# Qlearning Agent
class QLearningAgent(Agent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, player_id=1):
        self.q_table = {} 
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.player_id = player_id  

    def get_q_value(self, state, action):
        """Returns the Q-value for a given state-action pair."""
        state_tuple = tuple(map(tuple, state))
        return self.q_table.get((state_tuple, action), 0)


    def select_action(self, state, possible_actions):
        """Choose an action based on ε-greedy strategy."""
        if not possible_actions:
            return None  

        if random.random() < self.epsilon:
            return random.choice(possible_actions)  # Exploration
        else:
            q_values = [self.get_q_value(state, action) for action in possible_actions]
            max_q_value = max(q_values)
            
            # Get all actions that have the maximum Q-value
            max_q_actions = [action for action, q_value in zip(possible_actions, q_values) if q_value == max_q_value]
            
            # Randomly select among the actions with the maximum Q-value
            return random.choice(max_q_actions)

    def update_q_table(self, state, action, reward, next_state, next_possible_actions):
        """Update the Q-table using the Bellman equation."""
        # Convert numpy array states to tuple of tuples
        state_tuple = tuple(map(tuple, state))
        next_state_tuple = tuple(map(tuple, next_state))

        # Find the maximum Q-value for the possible actions in the next state
        max_future_q = max([self.get_q_value(next_state_tuple, next_action) for next_action in next_possible_actions], default=0)
        
        # Get the current Q-value and calculate the new Q-value using the Bellman equation
        current_q = self.get_q_value(state_tuple, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        
        # Update the Q-table
        self.q_table[(state_tuple, action)] = new_q

# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================


def train_qagent_with_custom_agent(num_games, qagent, opponent_agent, evaluation_function, debug=False):
    log_filename = f"{TRAINING_LOG_BASENAME}_with_custom_agent.log"
    if debug:
        logging.basicConfig(filename=log_filename, level=logging.INFO, filemode='w', format='%(message)s')

    game = CheckersGame(8)
    for episode in range(num_games):
        if episode % EVALUATION_INTERVAL == 0:
            evaluation_function(qagent, episode)
 

        state = game.reset()
        move_count = 0
        while game.game_winner() == 0:
            state = game.get_state()
            possible_actions = game.get_legal_moves(qagent.player_id)
            action = qagent.select_action(state, possible_actions)
            if action is None:
                break

            next_state, reward = game.perform_action_and_evaluate(action, qagent.player_id)
            next_possible_actions = game.get_legal_moves(qagent.player_id)
            qagent.update_q_table(state, action, reward, next_state, next_possible_actions)
            
            move_count += 1  

            possible_actions = game.get_legal_moves(opponent_agent.player_id)
            action = opponent_agent.select_action(game)
            if action is None:
                break

            next_state, reward = game.perform_action_and_evaluate(action, opponent_agent.player_id)
            
            move_count += 1  
            
        if debug:
            logging.info(f"Game {episode + 1}: Total moves played = {move_count}")
    
        
    q_table_filename = f"{Q_TABLE_SAVE_BASENAME}_with_custom_agent.json"
    save_q_table(qagent.q_table, q_table_filename)



def train_qagent_against_itself(num_games, qagent, evaluation_function, debug=False):
    log_filename = f"{TRAINING_LOG_BASENAME}_self_play.log"
    if debug:
        logging.basicConfig(filename=log_filename, level=logging.INFO, filemode='w', format='%(message)s')

    game = CheckersGame(8)
    for episode in range(num_games):
        if episode % EVALUATION_INTERVAL == 0:
            evaluation_function(qagent, episode)

        if debug:
            logging.info(f"Self-play Episode {episode + 1}")

        state = game.reset()
        move_count = 0
        while game.game_winner() == 0:
            for player_id in [qagent.player_id, -qagent.player_id]:
                state = game.get_state()
                possible_actions = game.get_legal_moves(player_id)
                action = qagent.select_action(state, possible_actions)
                if action is None:
                    break

                next_state, reward = game.perform_action_and_evaluate(action, player_id)
                next_possible_actions = game.get_legal_moves(player_id)
                qagent.update_q_table(state, action, reward, next_state, next_possible_actions)

                move_count += 1

                if game.game_winner() != 0:
                    break

        if debug:
            logging.info(f"Game {episode + 1}: Total moves played = {move_count}")
        q_table_filename = f"{Q_TABLE_SAVE_BASENAME}_self_play.json"
        save_q_table(qagent.q_table, q_table_filename)



    save_q_table(qagent.q_table, "./ignored_files/self_play_q_table.json")

def evaluate_agent(game, qagent, opponent_agent, num_games, log_filename):
    wins = 0
    logging.basicConfig(filename=log_filename, level=logging.INFO, filemode='w', format='%(message)s')

    for i in range(num_games):
        state = game.reset()
        while game.game_winner() == 0:
            action = qagent.select_action(state, game.get_legal_moves(qagent.player_id))
            state, _ = game.perform_action_and_evaluate(action, qagent.player_id)
            if game.game_winner() != 0:
                break

            action = opponent_agent.select_action(game)
            state, _ = game.perform_action_and_evaluate(action, opponent_agent.player_id)

        if game.game_winner() == qagent.player_id:
            wins += 1
        logging.info(f"Game {i + 1}: Winner - {'QAgent' if game.game_winner() == qagent.player_id else 'Opponent'}")

    return wins


def periodic_evaluation(qagent, episode, evaluation_results):
    game = CheckersGame(8)
    random_agent = RandomAgent(player_id=-1)
    alphabeta_agent = AlphaBetaAgent(player_id=-1)

    log_filename_random = f"{PERIODIC_EVALUATION_LOG_BASENAME}_ep{episode}_vs_random.log"
    log_filename_alphabeta = f"{PERIODIC_EVALUATION_LOG_BASENAME}_ep{episode}_vs_alphabeta.log"

    wins_against_random = evaluate_agent(game, qagent, random_agent, EVALUATION_GAMES, log_filename_random)
    wins_against_alphabeta = evaluate_agent(game, qagent, alphabeta_agent, EVALUATION_GAMES, log_filename_alphabeta)

    # Store results in a dictionary
    evaluation_results[episode] = {
        "wins_against_random": wins_against_random,
        "wins_against_alphabeta": wins_against_alphabeta
    }


def train_and_evaluate():
    qagent_random = QLearningAgent(player_id=1)
    qagent_alphabeta = QLearningAgent(player_id=1)
    qagent_self = QLearningAgent(player_id=1)

    evaluation_results = {
        "random": {},
        "alphabeta": {},
        "self": {}
    }
    # Adjust the lambda functions to pass both 'qagent' and 'episode'
    train_qagent_with_custom_agent(NUM_EPISODES, qagent_random, RandomAgent(player_id=-1), 
                                   lambda qagent, ep: periodic_evaluation(qagent, ep, evaluation_results["random"]), debug=True)
    train_qagent_with_custom_agent(NUM_EPISODES, qagent_alphabeta, AlphaBetaAgent(player_id=-1), 
                                   lambda qagent, ep: periodic_evaluation(qagent, ep, evaluation_results["alphabeta"]), debug=True)
    train_qagent_against_itself(NUM_EPISODES, qagent_self, 
                                lambda qagent, ep: periodic_evaluation(qagent, ep, evaluation_results["self"]), debug=True)

    # Save the accumulated evaluation results
    with open(EVALUATION_RESULTS_FILENAME, 'w') as file:
        json.dump(evaluation_results, file)

    # Save the Q-tables
    save_q_table(qagent_random.q_table, f'{Q_TABLE_SAVE_BASENAME}_vs_random.json')
    save_q_table(qagent_alphabeta.q_table, f'{Q_TABLE_SAVE_BASENAME}_vs_alphabeta.json')
    save_q_table(qagent_self.q_table, f'{Q_TABLE_SAVE_BASENAME}_vs_self.json')

train_and_evaluate()





# qagent = QLearningAgent(player_id=1)

# # Call the self-training function with the debug flag set to True or False as needed
# train_qagent_against_itself(10, qagent, debug=True)

# qagent = QLearningAgent(player_id=1)
# opponent_agent = RandomAgent(player_id=-1)

# # Call the function with the debug flag set to True or False as needed
# train_qagent_with_custom_agent(10, qagent, opponent_agent, debug=True)

# qagent = QLearningAgent(player_id=1)
# opponent_agent = RandomAgent(player_id=-1)

# train_qagent_with_custom_agent(10, qagent, opponent_agent)
