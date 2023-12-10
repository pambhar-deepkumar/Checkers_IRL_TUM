import numpy as np
from checkers_game import CheckersGame

def test_checkers_game():
    
    
    def test_board_initialization(self):
        # Test for board size 6
        game_6 = CheckersGame(board_size=6)
        self.assertEqual(game_6.board_size, 6)
        self.assertTrue((game_6.board[:2, 1::2] == 1).all())  # Check player 1's pieces
        self.assertTrue((game_6.board[4:, 1::2] == -1).all())  # Check player -1's pieces
        self.assertTrue((game_6.board[2:4, :] == 0).all())  # Check empty middle rows

        # Test for board size 8
        game_8 = CheckersGame(board_size=8)
        self.assertEqual(game_8.board_size, 8)
        self.assertTrue((game_8.board[:3, 1::2] == 1).all())  # Check player 1's pieces
        self.assertTrue((game_8.board[5:, 1::2] == -1).all())  # Check player -1's pieces
        self.assertTrue((game_8.board[3:5, :] == 0).all())  # Check empty middle rows

        # Test for board size 10
        game_10 = CheckersGame(board_size=10)
        self.assertEqual(game_10.board_size, 10)
        self.assertTrue((game_10.board[:4, 1::2] == 1).all())  # Check player 1's pieces
        self.assertTrue((game_10.board[6:, 1::2] == -1).all())  # Check player -1's pieces
        self.assertTrue((game_10.board[4:6, :] == 0).all())  # Check empty middle rows

    def test_invalid_board_size(self):
        # Test for invalid board size
        with self.assertRaises(ValueError):
            CheckersGame(board_size=7) 
            
    game = CheckersGame()
    assert game.board_size == 8
    assert game.player == 1
    assert np.array_equal(game.board, np.array([
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, -1, 0, -1, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, -1],
        [-1, 0, -1, 0, -1, 0, -1, 0]
    ]))

    # Test possible_pieces
    assert game.available_pieces_of_the_player(1) == [(0, 1), (0, 3), (0, 5), (0, 7), (1, 0), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (2, 5), (2, 7)]
    assert game.available_pieces_of_the_player(-1) == [(5, 0), (5, 2), (5, 4), (5, 6), (6, 1), (6, 3), (6, 5), (6, 7), (7, 0), (7, 2), (7, 4), (7, 6)]

    # Test possible_actions
    print(game.possible_actions(1))
    assert game.possible_actions(1) == [(2, 1, 3, 0), (2, 1, 3, 2), (2, 3, 3, 2), (2, 3, 3, 4), (2, 5, 3, 4), (2, 5, 3, 6), (2, 7, 3, 6)]
    assert game.possible_actions(-1) == [(5,0,4,1), (5,2,4,1), (5,2,4,3), (5,4,4,3), (5,4,4,5), (5,6,4,5), (5,6,4,7)]

    # Test get_piece
    action = (1, 0, 3, 3)
    game.get_piece(action)
    assert np.array_equal(game.board, np.array([
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, -1, 0, -1, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, -1],
        [-1, 0, -1, 0, -1, 0, -1, 0]
    ]))

    game.board = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, -1, 0, -1, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, -1],
        [-1, 0, -1, 0, -1, 0, -1, 0]
    ])
    # Test step
    action = (0, 1, 1, 0)
    player = 1
    reward = game.step(action, player)
    print(game.board)
    assert np.array_equal(game.board, np.array([
        [0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, -1, 0, -1, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, -1],
        [-1, 0, -1, 0, -1, 0, -1, 0]
    ]))
    assert reward == 0

    action = (0, 1, 1, 0)
    player = -1
    reward = game.step(action, player)
    assert np.array_equal(game.board, np.array([
        [0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, -1, 0, -1, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, -1],
        [-1, 0, -1, 0, -1, 0, -1, 0]
    ]))
    assert reward == 0

test_checkers_game()