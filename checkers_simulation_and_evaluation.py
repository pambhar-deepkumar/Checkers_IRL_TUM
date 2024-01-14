from checkers_model import CheckersGame
from Agents.minimax_agent import MinimaxAgent
from Agents.random_agent import RandomAgent 
import numpy as np
class CheckersTrainingAndEvaluationSimulator:
    def __init__(self, agent1, agent2, board_size=8):
        """
        Initialize the simulator with two agents and a board size.

        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.
            board_size (int): Size of the checkers board.
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.board_size = board_size

    def train_agents(self, num_games):
        """
        Train both agents by playing a specified number of games.

        Args:
            num_games (int): The number of games for training.
        """
        for _ in range(num_games):
            game = CheckersGame(board_size=self.board_size)
            current_player = 1

            while game.game_winner() == 0:
                agent = self.agent1 if current_player == 1 else self.agent2
                action = agent.select_action(game)
                if action is not None:
                    new_board, reward = game.perform_action_and_evaluate(action, current_player)
                    agent.learn_from_experience(game, action, reward, new_board)
                
                current_player *= -1  # Switch player

    def evaluate_agents(self, num_games, log_file_path="evaluation_log.txt", log=False):
        """
        Evaluate the performance of the agents by playing a specified number of games.

        Args:
            num_games (int): The number of games for evaluation.
            log_file_path (str): Path to the log file.

        Returns:
            dict: A dictionary with the results of the evaluations.
        """
        results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
        with open(log_file_path, 'w') as log_file:
            for game_number in range(num_games):
                print(f"Game {game_number + 1} started")
                if log:
                    log_file.write(f"Game {game_number + 1} started\n")
                game = CheckersGame(board_size=self.board_size)
                current_player = 1

                while game.game_winner() == 0:
                    if log:
                        log_file.write(game.render())
                        log_file.write(f"Player {current_player}'s turn\n")
                        log_file.write(f"Available actions: {game.generate_legal_moves(current_player)}\n")
                    agent = self.agent1 if current_player == 1 else self.agent2
                    action = agent.select_action(game)
                    if log:
                        log_file.write(f"Action taken: {action}\n")
                    if action is not None:
                        game.perform_action_and_evaluate(action, current_player)

                    current_player *= -1  # Switch player

                if log:
                    log_file.write(game.render())

                winner = game.game_winner()
                if winner == 1:
                    results['agent1_wins'] += 1
                elif winner == -1:
                    results['agent2_wins'] += 1
                else:
                    results['draws'] += 1
                if log:
                    if np.sum(game.board<0) == 0:
                        log_file.write(f"Player {winner} wins!\n Because all the pieces of the opponent are captured\n")
                    elif np.sum(game.board>0) == 0:
                        log_file.write(f"Player {winner} wins!\n Because all the pieces of the opponent are captured\n")
                    elif len(game.generate_legal_moves(-1)) == 0:
                        log_file.write(f"Player {winner} wins!\n Because the opponent has no possible actions\n")
                    elif len(game.generate_legal_moves(1)) == 0:
                        log_file.write(f"Player {winner} wins!\n Because the opponent has no possible actions\n")
                    log_file.write(f"Game {game_number + 1} ended\n")
                    log_file.write(f"Stats so far: {results}\n\n")
        return results


def main():
    agent1 = RandomAgent(player_id=1)
    agent2 = MinimaxAgent(player_id=-1)


    simulator = CheckersTrainingAndEvaluationSimulator(agent1, agent2, board_size=6)

    results = simulator.evaluate_agents(num_games=10)
    print(results)
    
if __name__ == "__main__":
    main()