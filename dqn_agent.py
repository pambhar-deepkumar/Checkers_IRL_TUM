import numpy as np 
import checker_env

def best_move(board, reinforced_model):
  compressed_board = checker_env.compress(board)
  boards = np.zeros((0, checker_env.COMPRESS_SIZE))
  boards = checker_env.generate_next(board)
  scores = reinforced_model.predict_on_batch(boards)
  max_index = np.argmax(scores)
  best = boards[max_index]
  return best

def print_board(board):
    checker_env.render(board)