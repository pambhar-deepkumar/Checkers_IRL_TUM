import json
import logging
from qlearningagent import QLearningAgent
from Agents.random_agent import RandomAgent
from Agents.alphabeta_agent import AlphaBetaAgent
from Assets.checkers_model import CheckersGame

def evaluate_agent(game, qagent, opponent_agent, num_games):
    wins = 0
    for _ in range(num_games):
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

def train_and_evaluate(num_episodes, evaluation_interval, final_evaluation_games):
    qagent = QLearningAgent(player_id=1)
    random_agent = RandomAgent(player_id=-1)
    alphabeta_agent = AlphaBetaAgent(player_id=-1)
    game = CheckersGame(8)

    evaluation_results = []

    for episode in range(1, num_episodes + 1):
        # Training phase
        # [Add your training code here]

        # Periodic evaluation
        if episode % evaluation_interval == 0 or episode == num_episodes:
            wins_against_random = evaluate_agent(game, qagent, random_agent, final_evaluation_games)
            wins_against_alphabeta = evaluate_agent(game, qagent, alphabeta_agent, final_evaluation_games)

            evaluation_results.append({
                "episode": episode,
                "wins_against_random": wins_against_random,
                "wins_against_alphabeta": wins_against_alphabeta
            })

            # Logging the intermediate results
            print(f"Evaluation after {episode} episodes: ")
            print(f"Wins against Random: {wins_against_random} / {final_evaluation_games}")
            print(f"Wins against AlphaBeta: {wins_against_alphabeta} / {final_evaluation_games}")

    # Save evaluation results and Q-table
    with open('evaluation_results.json', 'w') as file:
        json.dump(evaluation_results, file)

    save_q_table(qagent.q_table, 'final_q_table.json')

# Parameters
NUM_EPISODES = 2500
EVALUATION_INTERVAL = 500
FINAL_EVALUATION_GAMES = 500

# Run the training and evaluation
train_and_evaluate(NUM_EPISODES, EVALUATION_INTERVAL, FINAL_EVALUATION_GAMES)
