import numpy as np
import gym

class ConnectX(gym.Env):
    def __init__(self, width = 7, height = 6, connect = 4, start_player = 1):
        self.width = width
        self.height = height
        self.connect = connect
        self.board = np.zeros((height, width), dtype = int)
        self.start_player = start_player
        self.player = self.start_player
        self.winner = 0

        self.illegal_moves = 0

        # Gym attributes
        self.action_space = gym.spaces.Discrete(width)
        self.observation_space = gym.spaces.Box(low = 0, high = 2, shape = (height, width), dtype = int)

    def set_board(self, board):
        self.board = board
    
    def get_board(self):
        return self.board
    
    def is_valid_column(self, col):
        # Placed coin outside of board
        if (col < 0 or col >= self.width):
            return False
        
        # Column is full
        if (self.board[0, col] != 0):
            return False
        return True

    
    def place_coin(self, col):
        # TODO: Remove this, it's handled in step
        # if (not self.is_valid_column(col)):
        #     print("Invalid column")
        #     return False
        
        # Simulate gravity
        for i in range(self.height - 1, -1, -1):
            if (self.board[i, col] == 0):
                self.board[i, col] = self.player
                return True
        return False
    
    def valid_positions(self, only_cols = True) -> list:
        if (only_cols):
            return [i for i in range(self.width) if self.is_valid_column(i)]
        
        valid_positions = []
        for i in range(self.width):
            for j in range(self.height - 1, -1, -1):
                if (self.board[j, i] == 0):
                    valid_positions.append((j, i))
                    break
        return valid_positions


    def _change_player(self):
        if (self.player == 1):
            self.player = 2
        else:
            self.player = 1

    def has_winner(self) -> int:        
        # Vertical win
        for i in range(self.height - self.connect + 1):
            for j in range(self.width):
                for player in range(1, 3):
                    if (self.board[i, j] == self.board[i + 1, j] == self.board[i + 2, j] == self.board[i + 3, j] == player):
                        print("Vertical win", player)
                        self.winner = player
                        return player
        
        # Horizontal win
        for i in range(self.height):
            for j in range(self.width - self.connect + 1):
                for player in range(1, 3):
                    if (self.board[i, j] == self.board[i, j + 1] == self.board[i, j + 2] == self.board[i, j + 3] == player):
                        print("Horizontal win", player)
                        self.winner = player
                        return player
        
        # Diagonal win
        k = self.connect - self.height
        for _ in range(self.height):
            diag = np.diag(self.board, k=k)
            diag_flipped = np.diag(np.fliplr(self.board), k=k)

            for i in range(len(diag) - self.connect + 1):
                for player in range(1, 3):
                    if (diag[i] == diag[i + 1] == diag[i + 2] == diag[i + 3] == player or diag_flipped[i] == diag_flipped[i + 1] == diag_flipped[i + 2] == diag_flipped[i + 3] == player):
                        print("Diagonal win", player)
                        self.winner = player
                        return player
            k += 1
        return 0
    
    def is_done(self) -> bool:
        if (self.has_winner() != 0):
            print(f"Game is done\nWinner: {self.winner}")
            return True
        if (np.count_nonzero(self.board == 0) == 0):
            print(f"Game is done\nDraw")
            return True
        return False
    
    def handle_illegal_move(self, action):
        print("Invalid action")
        self.illegal_moves += 1
        info = {
            'illegal_moves': self.illegal_moves,
            'player': self.player,
            'legal_move': False
        }
        
        # Return a negative reward to the player who made repeated illegal moves
        if (self.illegal_moves >= 2):
            print("Too many invalid moves!")
            reward = -1
            return self.board, reward, True, info

        is_done = self.is_done()
        reward = self.calculate_reward()
        self.place_coin(action)

        return self.board, self.winner, is_done, info

    def step(self, action):
        """
        Args: action
        Returns: observation, reward, done, info
        """
        if (not self.is_valid_column(action)):
            return self.handle_illegal_move(action)

        self.illegal_moves = 0
        self.place_coin(action)
        is_done = self.is_done()
        reward = self.calculate_reward()
        info = {
            'illegal_moves': self.illegal_moves,
            'player': self.player,
            'legal_move': True
        }
        
        self._change_player()
        return self.board, reward, is_done, info
    
    def calculate_reward(self):
        if (self.winner == self.player):
            return 1
        elif (self.winner == 0):
            return 0
        else:
            return -1

    def render(self, mode = 'human'):
        for i in range(self.height):
            print("|", end = "")
            for j in range(self.width):
                if (self.board[i, j] == 1):
                    print("X", end = " ")
                elif (self.board[i, j] == 2):
                    print("O", end = " ")
                else:
                    print("-", end = " ")
            print("|\n", end="")
        print("\n")
    
    def reset(self):
        """
        Returns: observation
        """
        self.board = np.zeros((self.height, self.width), dtype = int)
        self.player = self.start_player
        self.winner = 0
        return self.board
    
    def close(self):
        pass

    def __str__(self) -> str:
        return '\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.board])