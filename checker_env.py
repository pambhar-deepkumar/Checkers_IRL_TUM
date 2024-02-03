import numpy as np

BOARD_SIZE = 8
PLAYER_PIECE = (BOARD_SIZE // 2 - 1) * BOARD_SIZE/2
COMPRESS_SIZE = (BOARD_SIZE * BOARD_SIZE) // 2

# TODO - Implemented 
def num_captured(board):
	return PLAYER_PIECE - np.sum(board < 0)

# TODO - Implemented 
def at_enemy(board):
    """Input Expnded board"""
    count = 0
    enemy_territory_start = (BOARD_SIZE // 2) + 1 
    for i in range(enemy_territory_start, BOARD_SIZE):
        count += np.sum(board[i] == 1)  
    return count
# TODO - Implemented
def at_middle(board):
    """Input Expnded board"""
    count = 0
    start = (BOARD_SIZE // 2)
    for i in range(start, start + 2):
        count += np.sum(board[i] == 1)
    return count
# TODO - Implemented
def num_men(board):
    """Input Expnded board"""
    return np.sum(board == 1)

# TODO - Implemented
def generate_branches(board, x, y):
    rows, cols = board.shape
    bb = np.array([compress(board)])  # Initialize with the current board state

    if board[x, y] >= 1 and x < rows - 2:
        temp_1 = board[x, y]
        # Right diagonal forward capture
        if y < cols - 2:
            if board[x+1, y+1] < 0 and board[x+2, y+2] == 0:
                board[x+2, y+2] = board[x, y]
                temp = board[x+1, y+1]
                board[x+1, y+1] = 0
                board[x, y] = 0  # Capture and move
                bb = np.vstack((bb, generate_branches(board, x+2, y+2)))
                board[x+1, y+1] = temp
                board[x, y] = temp_1
                board[x+2, y+2] = 0
                
        # Left diagonal forward capture
        if y > 1:
            if board[x+1, y-1] < 0 and board[x+2, y-2] == 0:
                board[x+2, y-2] = board[x, y]
                temp = board[x+1, y-1]
                board[x+1, y-1] = 0
                board[x, y] = 0  # Capture and move
                bb = np.vstack((bb, generate_branches(board, x+2, y-2)))
                board[x+1, y-1] = temp
                board[x, y] = temp_1
                board[x+2, y-2] = 0

    return bb
# TODO - Implemented
def np_board():
    return compress(init_board())

# TODO - Implemented
def generate_next(board):
    rows, cols = board.shape
    bb = np.array([compress(board)])  # Initialize with the current board state

    for i in range(rows):
        for j in range(cols):
            if board[i, j] > 0:
                generated_branches = generate_branches(board, i, j)[1:]  # Skip the initial state
                if generated_branches.size > 0:
                    bb = np.vstack((bb, generated_branches))
    
    # Simple move logic for non-king pieces
    for i in range(rows):
        for j in range(cols):
            if board[i, j] >= 1 and i < rows - 1:
                temp = board[i, j]
                # Right simple move
                if j < cols - 1 and board[i+1, j+1] == 0:
                    board[i+1, j+1] = board[i, j]
                    board[i, j] = 0
                    bb = np.vstack((bb, compress(board)))
                    board[i, j] = temp
                    board[i+1, j+1] = 0
                # Left simple move
                if j > 0 and board[i+1, j-1] == 0:
                    board[i+1, j-1] = board[i, j]
                    board[i, j] = 0
                    bb = np.vstack((bb, compress(board)))
                    board[i, j] = temp
                    board[i+1, j-1] = 0

    return bb[1:]  # Return all but the initial state

# TODO - Implemented
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

# TODO - Implemented
def available_pieces_of_the_player(board):
    return [(i, j) for i, row in enumerate(board)
            for j, value in enumerate(row) if value == 1]

# TODO - Implemented
def is_valid_position(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


# TODO - Implemented
def possible_actions(board):
    actions = []
    starters = available_pieces_of_the_player(board)
    directions = [(1, -1), (1, 1)] 

    for x, y in starters:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid_position(nx, ny) and board[nx][ny] == 0:
                actions.append((x, y, nx, ny))
            elif is_valid_position(nx, ny) and board[nx][ny] == -1:
                jx, jy = x + 2 * dx, y + 2 * dy
                if is_valid_position(jx, jy) and board[jx][jy] == 0:
                    actions.append((x, y, jx, jy))
    return actions

# TODO - Implemented
def capturables(board ):
    """Input Expnded board"""
    actions = possible_actions(board)
    captures = [action for action in actions if abs(action[0] - action[2]) > 1]  # Assuming action format is (x1, y1, x2, y2)
    return len(captures)

# TODO - Implemented
def semicapturables(board):
    """Input Expnded board"""
    return (PLAYER_PIECE - uncapturables(board) - capturables(reverse(board)))


def _is_uncapturable(board, x, y):
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

# TODO - Implemented
def uncapturables(board):
    """Count the number of player's pieces that can't be captured."""
    count = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if _is_uncapturable(board, i, j):
                count += 1
    return count


def reverse(board):
	b = -board
	b = np.fliplr(b)
	b = np.flipud(b)
	return b



def step(board, action):
    row1, col1, row2, col2 = action
    reward = 0
    game_status = 0  # Game is not over
    if action in possible_actions(board):
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
# TODO - Implemented
def game_winner(board):
    if np.sum(board < 0) == 0:
        return 1
    elif np.sum(board > 0) == 0:
        return -1
    elif not possible_actions(board):
        return 1
    elif not possible_actions(reverse(board)):
        return -1
    else:
        return 0
# TODO - Implemented
def render(board):
    game_state = " +" + "---+" * board.shape[0] + "\n"
    for row in board:
        game_state += " | " + " | ".join("0" if square == 1 else "X" if square == -1 else " "
                                        for square in row) + " |\n"
        game_state += " +" + "---+" * board.shape[0] + "\n"
    return game_state

# TODO - Implemented 
def possible_moves(board):
    return len(possible_actions(board))
# TODO - Implemented
def get_metrics(board):

    b = expand(board) # Assuming this prepares the board for analysis.
    capped = num_captured(b)
    potential = possible_moves(b) - possible_moves(reverse(b))
    men = num_men(b) - num_men(reverse(b))
    caps = capturables(b) - capturables(reverse(b))
    semicaps = semicapturables(b)
    uncaps = uncapturables(b) - uncapturables(reverse(b))
    mid = at_middle(b) - at_middle(reverse(b))
    far = at_enemy(b) - at_enemy(reverse(b))
    won = game_winner(b)
    score = 4*capped + potential + men + caps + 2*semicaps + 3*uncaps + 2*mid + 3*far + 100*won
    if score < 0:
        label = -1
    else:
        label = 1
    return np.array([label, capped, potential, men, caps, semicaps, uncaps, mid, far, won])

# TODO - Implemented the compress function
def compress(board):
    rows, cols = board.shape  # Get dimensions of the board
    # Calculate the total number of relevant squares (half of all squares for a checkerboard)
    num_relevant_squares = (rows * cols) // 2
    # Initialize the compressed array with the correct size
    b = np.zeros((num_relevant_squares), dtype='b')
    
    index = 0  # Start index for filling the compressed array
    for i in range(rows):
        for j in range(cols):
            # Select squares based on the checkerboard pattern
            if (i + j) % 2 == (1 if rows % 2 == 0 else 0):
                # This condition ensures we're selecting the correct squares
                # Adjusted to work with both even and odd number of columns
                b[index] = board[i, j]
                index += 1
                
    return b
# TODO - Implemented the expand function
def expand(compressed):
    original_rows = BOARD_SIZE
    original_cols = BOARD_SIZE
    # Initialize the expanded board with zeros
    b = np.zeros((original_rows, original_cols), dtype='b')
    index = 0  # Index for reading from the compressed array
    
    for i in range(original_rows):
        for j in range(original_cols):
            # Place compressed values back based on the checkerboard pattern
            if (i + j) % 2 == (1 if original_rows % 2 == 0 else 0):
                b[i, j] = compressed[index]
                index += 1
                
    return b


# board = init_board()
# print(board)

# p
# print(compress(reverse(board)))
# print(expand(compress(reverse(board))))


# start_board = expand(np_board())
# # print(start_board)
# boards_list = generate_next(start_board)
# # boards_list = generate_next(reverse(expand(boards_list[len(boards_list) - 1])))


# for board in boards_list:
#     print(expand(board))
#     print(get_metrics(board))
#     print("*******************")

