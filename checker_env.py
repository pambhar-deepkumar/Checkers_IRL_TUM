import numpy as np
BOARD_SIZE = 8
PLAYER_PIECE = (BOARD_SIZE // 2 - 1) * BOARD_SIZE/2
def num_captured(board):
	return PLAYER_PIECE - np.sum(board < 0)
def at_enemy(board):
    count = 0
    enemy_territory_start = BOARD_SIZE / 2 + 1 
    for i in range(enemy_territory_start, BOARD_SIZE):
        count += np.sum(board[i] == 1)  
    return count

def at_middle(board):
    count = 0
    start = (BOARD_SIZE / 2)
    for i in range(start, start + 2):
        count += np.sum(board[i] == 1)
    return count

def num_men(board):
	return np.sum(board == 1)


def init_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if (i + j) % 2 == 1:
                if i < (BOARD_SIZE // 2 - 1):
                    board[i][j] = 1  # Player 1's pieces
                elif i > (BOARD_SIZE // 2):
                    board[i][j] = -1  # Player -1's pieces
    return board

def available_pieces_of_the_player(board):
    return [(i, j) for i, row in enumerate(board)
            for j, value in enumerate(row) if value == 1]

def is_valid_position(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

def possible_actions(board):
    actions = []
    starters = available_pieces_of_the_player(board)
    directions = [(1, -1), (1, 1)] 

    for x, y in starters:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid_position(nx, ny, BOARD_SIZE) and board[nx][ny] == 0:
                actions.append((x, y, nx, ny))
            elif is_valid_position(nx, ny, BOARD_SIZE) and board[nx][ny] == -1:
                jx, jy = x + 2 * dx, y + 2 * dy
                if is_valid_position(jx, jy, BOARD_SIZE) and board[jx][jy] == 0:
                    actions.append((x, y, jx, jy))
    return actions

def count_capturable_pieces(board ):
    actions = possible_actions(board)
    captures = [action for action in actions if abs(action[0] - action[2]) > 1]  # Assuming action format is (x1, y1, x2, y2)
    return len(captures)

def is_supported(board, x, y):
    """Check if the piece at (x, y) has support (a friendly piece nearby)."""
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in directions:
        if 0 <= x + dx < len(board) and 0 <= y + dy < len(board):
            if board[x + dx][y + dy] == 1:  # Found a supporting piece
                return True
    return False

def semicapturables(board):
    """Calculate the number of player's pieces with at least one support."""
    count = 0
    for x in range(len(board)):
        for y in range(len(board)):
            if board[x][y] == 1:
                if is_supported(board, x, y, 1):
                    count += 1
    return count

import numpy as np

def is_uncapturable(board, x, y):
    """Check if a piece at position (x, y) is uncapturable."""
    if board[x][y] != 1:  # Only check for the player's own pieces
        return False
    
    # Edge pieces are considered safer but not necessarily uncapturable
    if x == 0 or x == len(board) - 1 or y == 0 or y == len(board) - 1:
        return True  # Simplification; actual uncapturability may depend on game rules
    
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in directions:
        # Check if the adjacent square in any direction is an opponent's piece
        # and the next square in the same direction is empty (indicating capturability)
        if 0 <= x + 2*dx < len(board) and 0 <= y + 2*dy < len(board):
            if board[x + dx][y + dy] == -1 and board[x + 2*dx][y + 2*dy] == 0:
                return False  # Can be captured by an opponent's piece
    return True  # No adjacent opponent's piece can capture it

def uncapturables(board):
    """Count the number of player's pieces that can't be captured."""
    count = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if is_uncapturable(board, i, j, 1):
                count += 1
    return count


def reverse(board):
	b = -board
	b = np.fliplr(b)
	b = np.flipud(b)
	return b

def compress_simple(board):
    # Flatten the board into a 1D array
    b = board.flatten()
    return b

def step(board, action):
    row1, col1, row2, col2 = action
    reward = 0
    game_status = 0  # Game is not over
    if action in possible_actions(board, 1, BOARD_SIZE):
        board[row1][col1] = 0
        board[row2][col2] = 1
        # Handle capture
        if abs(row2 - row1) > 1:
            captured_row = (row1 + row2) // 2
            captured_col = (col1 + col2) // 2
            board[captured_row][captured_col] = 0
            reward = 0.5  # Capturing gives a reward
        else:
            reward = -0.5  # Moving without capturing gives a slight penalty
        
        game_status = game_winner(board)
        if game_status == 1:
            reward += 1  # Winning gives a reward
        elif game_status == -1:
            reward -= 1  # Losing gives a penalty

    return board, reward, game_status

def game_winner(board):
    if np.sum(board < 0) == 0:
        return 1
    elif np.sum(board > 0) == 0:
        return -1
    elif not possible_actions(board, 1, board):
        return 1
    elif not possible_actions(board, 1, reverse(board)):
        return -1
    else:
        return 0

def render(board):
    game_state = " +" + "---+" * board.shape[0] + "\n"
    for row in board:
        game_state += " | " + " | ".join("0" if square == 1 else "X" if square == -1 else " "
                                        for square in row) + " |\n"
        game_state += " +" + "---+" * board.shape[0] + "\n"
    return game_state


def generate_captures(board, x, y):
    captures = []
    directions = [(1, -1), (1, 1)] 
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        nx2, ny2 = x + 2*dx, y + 2*dy
        if 0 <= nx2 < board.shape[0] and 0 <= ny2 < board.shape[1]:
            if board[nx, ny] == -1 and board[nx2, ny2] == 0:
                # Execute capture
                new_board = np.copy(board)
                new_board[x, y] = 0
                new_board[nx, ny] = 0
                new_board[nx2, ny2] = 1
                captures.append(new_board)
                # Recursively generate further captures from the new position
                captures.extend(generate_captures(new_board, nx2, ny2, 1))
    return captures

def generate_simple_moves(board, x, y):
    moves = []
    directions = [(1, -1), (1, 1)] 
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1] and board[nx, ny] == 0:
            new_board = np.copy(board)
            new_board[x, y] = 0
            new_board[nx, ny] = 1
            moves.append(new_board)
    return moves

def generate_next(board):
    next_boards = []
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            if board[x, y] == 1:
                next_boards.extend(generate_captures(board, x, y))
                next_boards.extend(generate_simple_moves(board, x, y))
    return next_boards

def possible_moves(board):
    return len(possible_actions(board, 1))

def get_metrics(board):
    # Assuming necessary auxiliary functions are defined elsewhere and work as expected.
    
    b = board # Assuming this prepares the board for analysis.

    # Metrics calculations
    capped = num_captured(b)
    potential = possible_moves(b) - possible_moves(reverse(b))
    men = num_men(b) - num_men(-b)
    caps = count_capturable_pieces(b) - count_capturable_pieces(reverse(b))
    semicaps = semicapturables(b)
    uncaps = uncapturables(b) - uncapturables(reverse(b))
    mid = at_middle(b) - at_middle(-b)
    far = at_enemy(b) - at_enemy(reverse(b))
    won = game_winner(b)

    # Scoring based on metrics
    score = 4*capped + potential + men + caps + 2*semicaps + 3*uncaps + 2*mid + 3*far + 100*won
    
    if score < 0:
        label = -1
    else:
        label = 1

    return np.array([label, capped, potential, men, caps, semicaps, uncaps, mid, far, won])


