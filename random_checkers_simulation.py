# random_checkers_simulation.py
import random
from checkers_model import CheckersGame
import numpy as np
def random_move(game, player):
    actions = game.generate_legal_moves(player)
    return random.choice(actions) if actions else None

def simulate_random_game(max_moves=1000):
    game = CheckersGame(board_size=8)
    move_count = 0
    current_player = 1
    with open('game_simulation_log.txt', 'w') as log_file:
        while move_count < max_moves and game.game_winner() == 0:
            available_actions = game.generate_legal_moves(current_player)
            action = random_move(game, current_player) if available_actions else None
            log_file.write(f"Available actions for Player {current_player}: {available_actions}\n")
            if action:
                _, reward = game.perform_action_and_evaluate(action, current_player)
                game_state = game.render()
                log_file.write(f"Player {current_player} performed action: {action}\n")
                log_file.write(f"Reward for the step: {reward}\n")
                log_file.write(game_state + "\n")
            else:
                log_file.write(f"No possible actions for Player {current_player}\n")
            current_player = -current_player  # Switch player
            move_count += 1

        winner = game.game_winner()
        if np.sum(game.board<0) == 0:
            log_file.write(f"Player {winner} wins!\n Because all the pieces of the opponent are captured\n")
        elif np.sum(game.board>0) == 0:
            log_file.write(f"Player {winner} wins!\n Because all the pieces of the opponent are captured\n")
        elif len(game.generate_legal_moves(-1)) == 0:
            log_file.write(f"Player {winner} wins!\n Because the opponent has no possible actions\n")
        elif len(game.generate_legal_moves(1)) == 0:
            log_file.write(f"Player {winner} wins!\n Because the opponent has no possible actions\n")
        

        if winner != 0:
            log_file.write(f"Player {winner} wins!\n")
        else:
            log_file.write("Game ended in a draw.\n")


if __name__ == "__main__":
    simulate_random_game()
