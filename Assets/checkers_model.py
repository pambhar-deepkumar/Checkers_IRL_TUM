import numpy as np

class CheckersGame:    
    def __init__(self, board_size=8):
        if board_size not in [6,8,10]:
            raise ValueError("Board size must be 6, 8 or 10")
        self.board_size = board_size
        self.reset()

    def reset(self):
        board = np.zeros((self.board_size, self.board_size))
        
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

    def get_legal_moves(self, player):
        """
        Generates a list of all legal moves available to the player.

        Args:
            player (int): The player number (1 or -1) for whom to generate moves.

        Returns:
            list: A list of tuples representing legal moves.
        """
        def is_valid_position(x, y):
            """
            This checks if the co ordinates are within the game board
            """
            return 0 <= x < self.board_size and 0 <= y < self.board_size

        actions = []
        starters = self.available_pieces_of_the_player(player)
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
        if abs(action[2] - action [0]) > 1:
            captured_row = (action[0] + action[2]) // 2
            captured_col = (action[1] + action[3]) // 2
            # jump
            self.board[captured_row][captured_col] = 0


    def game_winner(self):
        if np.sum(self.board<0) == 0:
            return 1
        elif np.sum(self.board>0) == 0:
            return -1
        elif len(self.get_legal_moves(-1)) == 0:
            return 1
        elif len(self.get_legal_moves(1)) == 0:
            return -1
        else:
            return 0

    def perform_action_and_evaluate(self, action, player):
        """Performs a step in the checkers game.

        Args:
            action (tuple): The action to be taken in the form (row1, col1, row2, col2).
            player (int): The player making the move. 1 for player 1, -1 for player 2.

        Returns:
            tuple: A tuple containing the updated game board and the reward obtained from the step.
        """
        reward = 0
        row1, col1, row2, col2 = action
        if action in self.get_legal_moves(player):
            self.board[row1][col1] = 0
            self.board[row2][col2] = player
            self.get_piece(action)
            
            if abs(row2 - row1) > 1:
                reward = 0.5  
            else:
                reward = -0.5  

            game_status = self.game_winner()
            if game_status == player:
                reward += 1  
            elif game_status == -player:
                reward -= 1  
            elif game_status == 0:
                reward += 0  
        return self.board, reward


    def get_state(self):
        return self.board   
    
    def render(self):
        game_state = " +" + "---+" * self.board_size + "\n"
        for row in self.board:
            game_state += " | " + " | ".join("0" if square == 1 else "X" if square == -1 else " "
                                            for square in row) + " |\n"
            game_state += " +" + "---+" * self.board_size + "\n"
        return game_state

