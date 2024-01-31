"""Only use with random, minimax and alphabeta agents"""


import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(current_dir, '../')
sys.path.insert(0, parent_dir)


from Assets.checkers_model import CheckersGame
from Agents.minimax_agent import MinimaxAgent
from Agents.random_agent import RandomAgent 
from Agents.alphabeta_agent import AlphaBetaAgent
import numpy as np
class CheckersSimulator:
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

    def simulate_games(self, num_games, log_file_path="evaluation_log.txt", log=False):
        """
        Evaluate the performance of the agents by playing a specified number of games.

        Args:
            num_games (int): The number of games for evaluation.
            log_file_path (str): Path to the log file.

        Returns:
            dict: A dictionary with the results of the evaluations.
        """
        results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
        total_moves = 0
        with open(log_file_path, 'w') as log_file:
            for game_number in range(num_games):
                num_moves = 0
                if log:
                    log_file.write(f"Game {game_number + 1} started\n")
                game = CheckersGame(board_size=self.board_size)
                current_player = 1

                while game.game_winner() == 0:
                    if log:
                        log_file.write(game.render())
                        log_file.write(f"Player {current_player}'s turn\n")
                        log_file.write(f"Available actions: {game.get_legal_moves(current_player)}\n")
                    agent = self.agent1 if current_player == self.agent1.player_id else self.agent2
                    action = agent.select_action(game)
                    if log:
                        log_file.write(f"Action taken: {action}\n")
                    if action is not None:
                        game.perform_action_and_evaluate(action, current_player)
                    num_moves += 1

                    current_player *= -1  # Switch player

                if log:
                    log_file.write(game.render())

                winner = game.game_winner()
                if winner == self.agent1.player_id:
                    results['agent1_wins'] += 1
                elif winner == self.agent2.player_id:
                    results['agent2_wins'] += 1
                else:
                    results['draws'] += 1
                if log:
                    if np.sum(game.board<0) == 0:
                        log_file.write(f"{'Agent1' if winner == self.agent1.player_id else 'Agent2'} wins!\n Because all the pieces of the opponent are captured\n")
                    elif np.sum(game.board>0) == 0:
                        log_file.write(f"{'Agent1' if winner == self.agent1.player_id else 'Agent2'} wins!\n Because all the pieces of the opponent are captured\n")
                    elif len(game.get_legal_moves(self.agent1.player_id)) == 0:
                        log_file.write(f"Agent 2 wins!\n Because the opponent has no possible actions\n")
                    elif len(game.get_legal_moves(self.agent2.player_id)) == 0:
                        log_file.write(f"Agent 1 wins!\n Because the opponent has no possible actions\n")
                    log_file.write(f"Game {game_number + 1} ended\n")
                    log_file.write(f"Stats so far: {results}\n")
                    log_file.write(f"Number of moves played: {num_moves}\n\n")
                    total_moves += num_moves
                    
            log_file.write(f"\n\n\n\nSimulation ended\n")
            log_file.write(f"Final stats: {results}\n")
            log_file.write(f"Total number of moves played: {total_moves}\n")
            log_file.write(f"Average number of moves played per game: {total_moves / num_games}\n")
        return results


def main():
    agent1 = MinimaxAgent(player_id=1, depth=4)
    # agent2 = AlphaBetaAgent(player_id=-1,depth=2)
    agent2 = RandomAgent(player_id=-1)
    simulator = CheckersSimulator(agent1, agent2, board_size=6)
    results = simulator.simulate_games(num_games=10,log_file_path="./ignored_files/logfile.txt",log = True)
    print(results)
    
if __name__ == "__main__":
    main()