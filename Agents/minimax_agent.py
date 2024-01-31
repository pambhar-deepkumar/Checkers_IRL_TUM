from .agent_base import Agent
import numpy as np
import copy
class MinimaxAgent(Agent):
    def __init__(self, player_id, depth=3):
        super().__init__(player_id)
        self.depth = depth

    def select_action(self, game):
        _, action = self.minimax(game, self.depth, self.player_id)
        return action

    def minimax(self, game, depth, player, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or game.game_winner() != 0:
            return self.evaluate(game), None

        best_move = None
        if player == self.player_id:
            max_eval = float('-inf')
            for move in game.get_legal_moves(player):
                new_game_state = self.simulate_move(game, move, player)
                eval = self.minimax(new_game_state, depth - 1, -player, alpha, beta)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in game.get_legal_moves(player):
                new_game_state = self.simulate_move(game, move, player)
                eval = self.minimax(new_game_state, depth - 1, -player, alpha, beta)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate(self, game):
        return np.sum(game.board == self.player_id) - np.sum(game.board == -self.player_id)

    def simulate_move(self, game, action, player):
        new_game_state = copy.deepcopy(game)
        new_game_state.perform_action_and_evaluate(action, player)
        return new_game_state

