import numpy as np

class CheckersGame:

    def __init__(self, board_size=8, player=1):
        if board_size not in [6,8,10]:
            raise ValueError("Board size must be 6, 8 or 10")
        self.board_size = board_size
        self.board = self.reset()
        self.player = player

    def reset(self):
        board = np.zeros((self.board_size, self.board_size))
        
        # place the pieces for player 1 (denoted by 1) and player -1 (denoted by -1)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + j) % 2 == 1:
                    if i < (self.board_size // 2 - 1):
                        board[i][j] = 1  # Player 1's pieces
                    elif i > (self.board_size // 2):
                        board[i][j] = -1  # Player -1's pieces
        
        self.board = board
        return board

    def available_pieces_of_the_player(self, player):
        """
        Returns a list of available pieces of the player
        Args:
            player (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [(i, j) for i, row in enumerate(self.board)
                for j, value in enumerate(row) if value == player]

    def possible_actions(self, player):
        """
        Returns a list of possible actions for the player

        Args:
            player (_type_): _description_
        """
        def is_valid_position(x, y):
            """
            This checks if the co ordinates are within the game board
            """
            return 0 <= x < self.board_size and 0 <= y < self.board_size

        actions = []
        starters = self.available_pieces_of_the_player(player)
        # Only forward movement are possible and hence the direction is determined by the player
        directions = [(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)]

        for x, y in starters:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # If valid and empty square, add the move
                if is_valid_position(nx, ny) and self.board[nx][ny] == 0:
                    actions.append((x, y, nx, ny))
                # If valid and enemy square, check if the jump is possible
                elif is_valid_position(nx, ny) and self.board[nx][ny] == -player:
                    #  If the jump is possible, add the move
                    jx, jy = x + 2 * dx, y + 2 * dy
                    if is_valid_position(jx, jy) and self.board[jx][jy] == 0:
                        actions.append((x, y, jx, jy))
        return actions

    def get_piece(self, action):
        if action[2] - action [0] > 1:
            captured_row = (action[0] + action[2]) // 2
            captured_col = (action[1] + action[3]) // 2
            # jump
            self.board[captured_row][captured_col] = 0


    def game_winner(self):
        if np.sum(self.board<0) == 0:
            return 1
        elif np.sum(self.board>0) == 0:
            return -1
        elif len(self.possible_actions(-1)) == 0:
            return -1
        elif len(self.possible_actions(1)) == 0:
            return 1
        else:
            return 0

    def step(self, action, player):
        row1, co1, row2, co2 = action
        if action in self.possible_actions(player):
            self.board[row1][co1] = 0
            self.board[row2][co2] = player
            self.get_piece(action)
            if self.game_winner() == player:
                reward = 1
            else:
                reward = 0
        else:
            reward = 0

        return reward

    def render(self):
        print("  +" + "---+" * self.board_size)
        for row in self.board:
            print(" | " + " | ".join("0" if square == 1 else "X" if square == -1 else " "
                                      for square in row) + " |")
            print("  +" + "---+" * self.board_size)

if __name__ == "__main__":
    game = CheckersGame(board_size=10)  
    game.render()
    print(game.possible_actions(1))
