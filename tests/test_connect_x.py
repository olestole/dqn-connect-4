from io import StringIO
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from connect_x.connect_x import ConnectX

class TestAdd(unittest.TestCase):
     def test_empty_board(self):
          game = ConnectX()
          board = game.board
          self.assertEqual(board.shape, (6, 7))
          self.assertEqual(board.dtype, int)
          self.assertEqual(board.size, 42)
     
     def test_vertical_win(self):
          capturedOutput = StringIO()                  # Create StringIO object
          sys.stdout = capturedOutput                  #  and redirect stdout.
          
          game = ConnectX()
          new_board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0]])
          game.set_board(new_board)
          is_done = game.is_done()
          sys.stdout = sys.__stdout__                   # Reset redirect.
          
          self.assertIn("Vertical win 1", capturedOutput.getvalue())
          self.assertEqual(is_done, True)
          self.assertEqual(game.winner, 1)
     
     def test_horizontal_win(self):
          capturedOutput = StringIO()                  # Create StringIO object
          sys.stdout = capturedOutput                  #  and redirect stdout.
          
          game = ConnectX()
          new_board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 1, 0, 0]])
          game.set_board(new_board)
          is_done = game.is_done()
          sys.stdout = sys.__stdout__                   # Reset redirect.
          
          self.assertIn("Horizontal win 1", capturedOutput.getvalue())
          self.assertEqual(is_done, True)
          self.assertEqual(game.winner, 1)
     
     def test_diagonal_win(self):
          capturedOutput = StringIO()                  # Create StringIO object
          sys.stdout = capturedOutput                  #  and redirect stdout.
          
          game = ConnectX()
          new_board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0]])
          game.set_board(new_board)
          is_done = game.is_done()
          sys.stdout = sys.__stdout__                   # Reset redirect.
          
          self.assertIn("Diagonal win 1", capturedOutput.getvalue())
          self.assertEqual(is_done, True)
          self.assertEqual(game.winner, 1)

     def test_draw(self):
          capturedOutput = StringIO()                  # Create StringIO object
          sys.stdout = capturedOutput                  #  and redirect stdout.
          
          game = ConnectX()
          new_board = np.array([[2, 1, 2, 2, 2, 1, 2],
                                [1, 2, 1, 1, 1, 2, 1],
                                [2, 1, 2, 2, 2, 1, 2],
                                [1, 2, 1, 1, 1, 2, 1],
                                [2, 1, 2, 2, 2, 1, 2],
                                [1, 2, 1, 1, 1, 2, 1]])
          game.set_board(new_board)
          is_done = game.is_done()
          sys.stdout = sys.__stdout__                   # Reset redirect.
          
          self.assertIn("Draw", capturedOutput.getvalue())
          self.assertEqual(is_done, True)
          self.assertEqual(game.winner, 0)
     
     def test_reset_board(self):
          game = ConnectX()
          new_board = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0]])
          game.set_board(new_board)
          
          is_done = game.is_done()
          self.assertEqual(is_done, True)
          self.assertEqual(game.winner, 1)

          game.reset()
          board = game.board
          
          self.assertEqual(board.shape, (6, 7))
          self.assertEqual(board.dtype, int)
          self.assertEqual(board.size, 42)
          self.assertEqual(board.sum(), 0)
          

if __name__ == '__main__':
    unittest.main()