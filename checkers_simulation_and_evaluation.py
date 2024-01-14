from checkers_model import CheckersGame
from Agents.minimax_agent import MinimaxAgent
from Agents.random_agent import RandomAgent 

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

    def evaluate_agents(self, num_games):
        """
        Evaluate the performance of the agents by playing a specified number of games.

        Args:
            num_games (int): The number of games for evaluation.

        Returns:
            dict: A dictionary with the results of the evaluations.
        """
        results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
        for _ in range(num_games):
            game = CheckersGame(board_size=self.board_size)
            current_player = 1

            while game.game_winner() == 0:
                agent = self.agent1 if current_player == 1 else self.agent2
                action = agent.select_action(game)
                if action is not None:
                    game.perform_action_and_evaluate(action, current_player)

                current_player *= -1  # Switch player

            winner = game.game_winner()
            if winner == 1:
                results['agent1_wins'] += 1
            elif winner == -1:
                results['agent2_wins'] += 1
            else:
                results['draws'] += 1

        return results

def main():
    agent1 = MinimaxAgent(player_id=1)
    agent2 = RandomAgent(player_id=-1)

    simulator = CheckersTrainingAndEvaluationSimulator(agent1, agent2, board_size=8)

    # Train agents
    simulator.train_agents(num_games=1)

    # Evaluate agents
    results = simulator.evaluate_agents(num_games=10)
    print(results)
    
if __name__ == "__main__":
    main()