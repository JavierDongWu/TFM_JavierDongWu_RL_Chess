import chess
import numpy as np

from stockfish import Stockfish
import chess_agents

mapper = {}
mapper["p"] = 0
mapper["r"] = 1
mapper["n"] = 2
mapper["b"] = 3
mapper["q"] = 4
mapper["k"] = 5
mapper["P"] = 0
mapper["R"] = 1
mapper["N"] = 2
mapper["B"] = 3
mapper["Q"] = 4
mapper["K"] = 5


class Board(object):

    def __init__(self, opposing_agent, FEN=None, capture_reward_factor=0.01, reward="greedy"):
        """
        Chess Board Environment
        Args:
            FEN: str
                Starting FEN notation, if None then start in the default chess position
            capture_reward_factor: float [0,inf]
                reward for capturing a piece. Multiply material gain by this number. 0 for normal chess.
        """
        self.FEN = FEN
        self.capture_reward_factor = capture_reward_factor
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.layer_board = np.zeros(shape=(8, 8, 8))
        self.init_layer_board()
        self.opposing_agent = opposing_agent
        self.reward = reward
        if reward == "minmax":
            self.minmaxagent = chess_agents.MinMaxAlphaBetaIncrementalEvaluationAgent(1) # Used only to evaluate rewards
        elif reward == "stockfish":
            self.stockfish = Stockfish(path="stockfish", depth=2, parameters={"Minimum Thinking Time": 1})
            self.previous_board = None
            self.previous_evaluation = 0
        print("Environment reward:", reward)

    def init_layer_board(self):
        """
        Initalize the numerical representation of the environment
        Returns:

        """
        self.layer_board = np.zeros(shape=(8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = mapper[piece.symbol()]
            self.layer_board[layer, row, col] = sign
            self.layer_board[6, :, :] = 1 / self.board.fullmove_number
        if self.board.turn:
            self.layer_board[6, 0, :] = 1
        else:
            self.layer_board[6, 0, :] = -1
        self.layer_board[7, :, :] = 1

    def update_layer_board(self, move=None):
        self._prev_layer_board = self.layer_board.copy()
        self.init_layer_board()

    def pop_layer_board(self):
        self.layer_board = self._prev_layer_board.copy()
        self._prev_layer_board = None

    def step(self, action, test=True):
        """
        Run a step
        Args:
            action: python chess move
        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: float
                Difference in material value after the move
        """
        if self.reward == "stockfish":
            #Stockfish reward based on the difference in evaluation value after a move
            color = self.board.turn
            evaluation_before = self.stockfish_evaluation(color)
            self.board.push(action)
            self.update_layer_board(action)
            evaluation_after = self.stockfish_evaluation(color)
            auxiliary_reward = (evaluation_after - evaluation_before) * 100 * self.capture_reward_factor
            #auxiliary_reward = (evaluation_after) * 100 * self.capture_reward_factor
        elif self.reward == "minmax":
            # Minmax reward based on the difference in evaluation value after a move
            board_copy = self.board.copy()
            self.minmaxagent.set_board(board_copy)
            if board_copy.turn:
                evaluation_before = self.minmaxagent.init_evaluate_board()/9999
            else:
                evaluation_before = -self.minmaxagent.init_evaluate_board()/9999
            evaluation_after = self.minmaxagent.predict_state(board_copy, action)
            self.board.push(action)
            self.update_layer_board(action)
            auxiliary_reward = (evaluation_after - evaluation_before) * 100 * self.capture_reward_factor
        else:
            piece_balance_before = self.get_material_value()
            self.board.push(action)
            self.update_layer_board(action)
            piece_balance_after = self.get_material_value()
            auxiliary_reward = (piece_balance_after - piece_balance_before) * self.capture_reward_factor
        result = self.board.result()
        if result == "*":
            reward = 0
            episode_end = False
        elif result == "1-0":
            reward = 1
            episode_end = True
        elif result == "0-1":
            reward = -1
            episode_end = True
        elif result == "1/2-1/2":
            reward = 0
            episode_end = True
        reward += auxiliary_reward

        return episode_end, reward

    def get_random_action(self):
        """
        Sample a random action
        Returns: move
            A legal python chess move.

        """
        legal_moves = [x for x in self.board.generate_legal_moves()]
        legal_moves = np.random.choice(legal_moves)
        return legal_moves

    def project_legal_moves(self):
        """
        Create a mask of legal actions
        Returns: np.ndarray with shape (64,64)
        """
        self.action_space = np.zeros(shape=(64, 64))
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        for move in moves:
            self.action_space[move[0], move[1]] = 1
        return self.action_space

    def get_material_value(self):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        pawns = 1 * np.sum(self.layer_board[0, :, :])
        rooks = 5 * np.sum(self.layer_board[1, :, :])
        minor = 3 * np.sum(self.layer_board[2:4, :, :])
        queen = 9 * np.sum(self.layer_board[4, :, :])
        return pawns + rooks + minor + queen

    def reset(self):
        """
        Reset the environment
        Returns:

        """
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_layer_board()

    #Evaluation function using the Stockfish engine
    def stockfish_evaluation(self, color):
        board_fen = self.board.fen()
        if self.stockfish.is_fen_valid(board_fen):
            self.stockfish.set_fen_position(board_fen)
            evaluation = self.stockfish.get_evaluation()
        else:
            print("fen no es valido", board_fen)
            if color == chess.WHITE:
                evaluation = {"type": "cp", "value": -10000}
            else:
                evaluation = {"type": "cp", "value": 10000}

        if evaluation["type"] == "cp":
            eval = evaluation["value"]
        else:
            if evaluation["value"] != 0:
                eval = 10000/evaluation["value"]
            else:
                #It should never happen
                print("value 0 jaque")
                exit()

        if color == chess.BLACK:
            eval = -eval
        eval = np.float64(eval/10000) 
        if eval > 1:
            eval = np.float64(0.9999)
        elif eval < -1:
            eval = np.float64(-0.9999)

        return eval