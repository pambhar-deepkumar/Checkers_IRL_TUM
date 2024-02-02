import numpy as np
import random
from Assets.checkers_model import CheckersGame
from Agents.random_agent import RandomAgent
from Agents.alphabeta_agent import AlphaBetaAgent
from Agents.qlearning_agent import QLearningAgent
from Agents.agent_base import Agent
import json
import setup

# Training parameters
NUM_EPISODES = 2
EVALUATION_INTERVAL = 2
EVALUATION_GAMES = 2
FINAL_EVALUATION_GAMES = 2
GAMEBOARD_SIZE = 8  
DEBUG = True
# ==================================================================
def save_q_table(q_table, filename):
    q_table_str_keys = {str(key): value for key, value in q_table.items()}
    with open(filename, 'w') as file:
        json.dump(q_table_str_keys, file)
        
def load_q_table(filename):
    with open(filename, 'r') as file:
        q_table_str_keys = json.load(file)
    q_table = {eval(key): value for key, value in q_table_str_keys.items()}
    return q_table

#====================================================================

def train_qagent_with_custom_agent(num_games, qagent, opponent_agent, evaluation_function):
    game = CheckersGame(GAMEBOARD_SIZE)
    for episode in range(num_games):
        if episode % EVALUATION_INTERVAL == 0:
            if DEBUG:
                print(f"Performing evaluation at episode {episode}")
            evaluation_function(qagent, episode)

        state = game.reset()
        while game.game_winner() == 0:
            state = game.get_state()
            possible_actions = game.possible_actions(qagent.player_id)
            action = qagent.select_action(state, possible_actions)
            if action is None:
                break

            next_state, reward = game.step(action, qagent.player_id)
            next_possible_actions = game.possible_actions(qagent.player_id)
            qagent.update_q_table(state, action, reward, next_state, next_possible_actions)
            
            possible_actions = game.possible_actions(opponent_agent.player_id)
            action = opponent_agent.select_action(game)
            if action is None:
                break

            next_state, reward = game.step(action, opponent_agent.player_id)
    if DEBUG:
        print(f"Saving q_table.")
    q_table_filename = f"{setup.Q_TABLE_SAVE_BASENAME}_with_custom_agent.json"
    save_q_table(qagent.q_table, q_table_filename)

def train_qagent_against_itself(num_games, qagent, evaluation_function):
    game = CheckersGame(GAMEBOARD_SIZE)
    for episode in range(num_games):
        if episode % EVALUATION_INTERVAL == 0:
            if DEBUG:
                print(f"Performing evaluation at episode {episode}")
            evaluation_function(qagent, episode)

        state = game.reset()
        while game.game_winner() == 0:
            for player_id in [qagent.player_id, -qagent.player_id]:
                state = game.get_state()
                possible_actions = game.possible_actions(player_id)
                action = qagent.select_action(state, possible_actions)
                if action is None:
                    break

                next_state, reward = game.step(action, player_id)
                next_possible_actions = game.possible_actions(player_id)
                qagent.update_q_table(state, action, reward, next_state, next_possible_actions)

                if game.game_winner() != 0:
                    break

    if DEBUG:
        print(f"Saving q_table.")
    q_table_filename = f"{setup.Q_TABLE_SAVE_BASENAME}_self_play.json"
    save_q_table(qagent.q_table, q_table_filename)


def evaluate_agent(game, qagent, opponent_agent, num_games):
    wins = 0

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

    return wins

def periodic_evaluation(qagent, episode, evaluation_results):
    game = CheckersGame(GAMEBOARD_SIZE)
    random_agent = RandomAgent(player_id=-1)
    alphabeta_agent = AlphaBetaAgent(player_id=-1)

    wins_against_random = evaluate_agent(game, qagent, random_agent, EVALUATION_GAMES)
    wins_against_alphabeta = evaluate_agent(game, qagent, alphabeta_agent, EVALUATION_GAMES)

    evaluation_results[episode] = {
        "wins_against_random": wins_against_random,
        "wins_against_alphabeta": wins_against_alphabeta
    }

def final_evaluation(qagent, evaluation_results, agent_type):
    game = CheckersGame(GAMEBOARD_SIZE)
    random_agent = RandomAgent(player_id=-1)
    alphabeta_agent = AlphaBetaAgent(player_id=-1)

    wins_against_random = evaluate_agent(game, qagent, random_agent, FINAL_EVALUATION_GAMES)
    wins_against_alphabeta = evaluate_agent(game, qagent, alphabeta_agent, FINAL_EVALUATION_GAMES)

    evaluation_results[f"final_{agent_type}"] = {
        "wins_against_random": wins_against_random,
        "wins_against_alphabeta": wins_against_alphabeta
    }

def train_and_evaluate():
    if DEBUG:
        print("Initializing agents")
    qagent_random = QLearningAgent(player_id=1)
    qagent_alphabeta = QLearningAgent(player_id=1)
    qagent_self = QLearningAgent(player_id=1)

    evaluation_results = {"random": {}, "alphabeta": {}, "self": {}}

    if DEBUG:
        print("Training against random agent")
    train_qagent_with_custom_agent(NUM_EPISODES, qagent_random, RandomAgent(player_id=-1), 
                                   lambda qagent, ep: periodic_evaluation(qagent, ep, evaluation_results["random"]))
    if DEBUG:
        print("Training against alphabeta agent")
    train_qagent_with_custom_agent(NUM_EPISODES, qagent_alphabeta, AlphaBetaAgent(player_id=-1), 
                                   lambda qagent, ep: periodic_evaluation(qagent, ep, evaluation_results["alphabeta"]))
    if DEBUG:
        print("Training against itself")
    train_qagent_against_itself(NUM_EPISODES, qagent_self, 
                                lambda qagent, ep: periodic_evaluation(qagent, ep, evaluation_results["self"]))

    if DEBUG:
        print("Final evaluation")
    final_evaluation(qagent_random, evaluation_results, "random")
    final_evaluation(qagent_alphabeta, evaluation_results, "alphabeta")
    final_evaluation(qagent_self, evaluation_results, "self")

    with open(setup.EVALUATION_RESULTS_FILENAME, 'w') as file:
        json.dump(evaluation_results, file)

    save_q_table(qagent_random.q_table, f'{setup.Q_TABLE_SAVE_BASENAME}_vs_random.json')
    save_q_table(qagent_alphabeta.q_table, f'{setup.Q_TABLE_SAVE_BASENAME}_vs_alphabeta.json')
    save_q_table(qagent_self.q_table, f'{setup.Q_TABLE_SAVE_BASENAME}_vs_self.json')

def main():
    train_and_evaluate()

if __name__ == "__main__":
    print("Starting training and evaluation")
    main()