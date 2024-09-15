from keras.layers import Input, Dense, Flatten, Concatenate, Conv2D, Dropout
from keras.losses import mean_squared_error
from keras.models import Model, clone_model, load_model
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np

from stockfish import Stockfish
from stockfish import StockfishException
import chess
import contextlib
import sys
import os
import random
import RLC_modified.RLC.real_chess.agent as agent_rlc
import RLC_modified.RLC.real_chess.environment as environment_rlc
import RLC_modified.RLC.real_chess.learn as learn_rlc
import RLC_modified.RLC.real_chess.tree as tree_rlc
import layer_board_processor
from sb3_contrib import MaskablePPO
from ppo_agent.chess_env import chess_utils
import ppo_agent.chess_env
import ppo_agent.chess_env.chess
import ppo_agent.one_agent_chess

"""
Implementation of chess players, each of them can have functions to predict a move given a board or 
to predict the state of a board
"""

#Function to suppress stdout temporarily
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

#Player that always moves randomly
class RandomAgent(object):
    def __init__(self):
        pass

    def predict_move(self, board):
        move = random.choice(list(board.legal_moves))
        return move

"""
Autor: Arjan Groen
Url: https://github.com/arjangroen/RLC/tree/master

This agent gives the prediction for the state of a board, based on the material values
"""
class GreedyAgent(object):
    def __init__(self, color=-1):
        self.color = color

    def predict(self, layer_board, noise=True):
        layer_board1 = layer_board[0, :, :, :]
        pawns = 1 * np.sum(layer_board1[0, :, :])
        rooks = 5 * np.sum(layer_board1[1, :, :])
        minor = 3 * np.sum(layer_board1[2:4, :, :])
        queen = 9 * np.sum(layer_board1[4, :, :])

        maxscore = 40
        material = pawns + rooks + minor + queen
        board_value = self.color * material / maxscore
        if noise:
            added_noise = np.random.randn() / 1e3

        return board_value + added_noise
    
#This player always tries to make a capture move, if it can not then it moves randomly
class CaptureAgent(object):
    def __init__(self):
        pass

    def predict_move(self, board):
        for move in board.legal_moves:
            if board.is_capture(move):
                return move
        
        move = random.choice(list(board.legal_moves))

        return move            


"""
Autor: Arjan Groen
Url: https://github.com/arjangroen/RLC/tree/master

Implementation to integrate the models with the format ".h5", comes with functions to predict the state and the move 
"""
class AgentRLCModel(object):
    def __init__(self, lr=0.003, network='big'):
        self.optimizer = RMSprop(learning_rate=lr)
        self.model = Model()
        self.proportional_error = False
        self.init_bignet()
        self.mapper = {}
        self.mapper["p"] = 0
        self.mapper["r"] = 1
        self.mapper["n"] = 2
        self.mapper["b"] = 3
        self.mapper["q"] = 4
        self.mapper["k"] = 5
        self.mapper["P"] = 0
        self.mapper["R"] = 1
        self.mapper["N"] = 2
        self.mapper["B"] = 3
        self.mapper["Q"] = 4
        self.mapper["K"] = 5
        self.model.load_weights('RLC_model.h5')

    def init_bignet(self):
        layer_state = Input(shape=(8, 8, 8), name='state')
        conv_xs = Conv2D(4, (1, 1), activation='relu')(layer_state)
        conv_s = Conv2D(8, (2, 2), strides=(1, 1), activation='relu')(layer_state)
        conv_m = Conv2D(12, (3, 3), strides=(2, 2), activation='relu')(layer_state)
        conv_l = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(layer_state)
        conv_xl = Conv2D(20, (8, 8), activation='relu')(layer_state)
        conv_rank = Conv2D(3, (1, 8), activation='relu')(layer_state)
        conv_file = Conv2D(3, (8, 1), activation='relu')(layer_state)

        f_xs = Flatten()(conv_xs)
        f_s = Flatten()(conv_s)
        f_m = Flatten()(conv_m)
        f_l = Flatten()(conv_l)
        f_xl = Flatten()(conv_xl)
        f_r = Flatten()(conv_rank)
        f_f = Flatten()(conv_file)

        dense1 = Concatenate(name='dense_bass')([f_xs, f_s, f_m, f_l, f_xl, f_r, f_f])
        dense2 = Dense(256, activation='sigmoid')(dense1)
        dense3 = Dense(128, activation='sigmoid')(dense2)
        dense4 = Dense(56, activation='sigmoid')(dense3)
        dense5 = Dense(64, activation='sigmoid')(dense4)
        dense6 = Dense(32, activation='sigmoid')(dense5)

        value_head = Dense(1)(dense6)

        self.model = Model(inputs=layer_state,
                           outputs=value_head)
        self.model.compile(optimizer=self.optimizer,
                           loss=mean_squared_error
                           )
    
    def get_layer_board(self, board):
        """
        Returns the numerical representation of the environment
        """
        layer_board = np.zeros(shape=(8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = self.mapper[piece.symbol()]
            layer_board[layer, row, col] = sign
            layer_board[6, :, :] = 1 / board.fullmove_number
        if board.turn:
            layer_board[6, 0, :] = 1
        else:
            layer_board[6, 0, :] = -1
        layer_board[7, :, :] = 1
        return layer_board
    
    def search_model_best_move(self, board):
        max_move = None
        turn = board.turn
        if turn == chess.WHITE:
            max_value = np.NINF
        else:
            max_value = np.inf
        board_copy = board.copy()
        for move in board_copy.legal_moves:
            board_copy.push(move)
            #env.step(move)
            #if env.board.result() == "0-1":
            #    max_move = move
            #    env.board.pop()
            #    env.init_layer_board()
            #    break
            if (board_copy.result() == "0-1" and board.turn == chess.BLACK) or (board_copy.result() == "1-0" and board.turn == chess.WHITE):
                max_move = move
                break
            state_value = self.predict_state(board_copy)
            if turn == chess.WHITE:
                if state_value > max_value:
                    max_move = move
                    max_value = state_value
            else:
                if state_value < max_value:
                    max_move = move
                    max_value = state_value
            board_copy.pop()

        return max_move

    def predict_state(self, board):
        board_layer = self.get_layer_board(board)
        board_layer_state = np.expand_dims(board_layer, axis=0)
        with suppress_stdout():
            state = self.model.predict(board_layer_state)
            
        return state
    
    def predict_move(self, board):
        return self.search_model_best_move(board)

#This player uses the Stockfish engine to predict the moves and states
class StockfishAgent:
    def __init__(self, depth=1, parameters = {}):
        self.stockfish = Stockfish(path="stockfish", depth=depth, parameters=parameters)
        self.processor = layer_board_processor.ChessBoardProcessor()

    def predict_move(self, board):
        self.stockfish.set_fen_position(board.fen())
        move_uci = self.stockfish.get_best_move()
        move = chess.Move.from_uci(move_uci)

        return move
    
    #Function used to predict the board evaluation, used for the black player
    def predict(self, layer_board):
        #Obtain the board after the made move, to predict the value 
        board = self.processor.reconstruct_board(layer_board)

        board_fen = board.fen()
        if self.stockfish.is_fen_valid(board_fen):
            self.stockfish.set_fen_position(board_fen)
            evaluation = self.stockfish.get_evaluation()
        else:
            print("fen no es valido", board_fen)
            # if board.turn == chess.WHITE:
            #     evaluation = {"type": "cp", "value": 10000/2}
            # else:
            #     evaluation = {"type": "cp", "value": -10000/2}
            evaluation = {"type": "cp", "value": 10000/2}

        if evaluation["type"] == "cp":
            eval = evaluation["value"]
        else:
            eval = (10000/2)/evaluation["value"]

        # if board.turn == chess.WHITE:
        #     eval = -eval

        #Always evaluating as black
        eval = -eval

        # if board.turn == chess.BLACK:
        #     print("board es black")
        # else:
        #     print("board es white")

        eval = np.float64(eval*2/10000) 
        if eval > 1:
            eval = np.float64(0.9999)
        elif eval < -1:
            eval = np.float64(-0.9999)
    
        return eval
    

"""
Autor: Andreas Stöckl
Url: https://github.com/astoeckl/mediumchess/blob/master/Blog2.ipynb

This player is the integration of the chess program using MinMax and AlphaBeta, it has functions to predict moves and states
"""
class MinMaxAlphaBetaIncrementalEvaluationAgent:
    def __init__(self, depth):
        self.pawntable = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10,-20,-20, 10, 10,  5,
            5, -5,-10,  0,  0,-10, -5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5,  5, 10, 25, 25, 10,  5,  5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0,  0,  0,  0,  0,  0,  0,  0]

        self.knightstable = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50]

        self.bishopstable = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20]

        self.rookstable = [
            0,  0,  0,  5,  5,  0,  0,  0,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            5, 10, 10, 10, 10, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0]

        self.queenstable = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  5,  5,  5,  5,  5,  0,-10,
            0,  0,  5,  5,  5,  5,  0, -5,
            -5,  0,  5,  5,  5,  5,  0, -5,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20]

        self.old_kingstable = [
            20, 30, 10,  0,  0, 10, 30, 20,
            20, 20,  0,  0,  0,  0, 20, 20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30]
        
        self.kingstable = [
            20, 30, 10,  0,  0, 10, 30, 20,
            20, 20,  0,  0,  0,  0, 20, 20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -5,-10,-10,-20,-20,-10,-10,-5,
            5,10,10,10,10,10,10,5,
            -30,-40,-40,-50,-50,-40,-40,-30]

        self.piecetypes = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING ]
        self.tables = [self.pawntable, self.knightstable, self.bishopstable, self.rookstable, self.queenstable, self.kingstable]
        self.piecevalues = [100,320,330,500,900]
        
        self.board = chess.Board()
        self.movehistory = []

        self.init_evaluate_board()

        self.processor = layer_board_processor.ChessBoardProcessor()

        self.depth = depth

    def init_evaluate_board(self):        
        wp = len(self.board.pieces(chess.PAWN, chess.WHITE))
        bp = len(self.board.pieces(chess.PAWN, chess.BLACK))
        wn = len(self.board.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(self.board.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(self.board.pieces(chess.BISHOP, chess.WHITE))
        bb = len(self.board.pieces(chess.BISHOP, chess.BLACK))
        wr = len(self.board.pieces(chess.ROOK, chess.WHITE))
        br = len(self.board.pieces(chess.ROOK, chess.BLACK))
        wq = len(self.board.pieces(chess.QUEEN, chess.WHITE))
        bq = len(self.board.pieces(chess.QUEEN, chess.BLACK))
        
        material = 100*(wp-bp)+320*(wn-bn)+330*(wb-bb)+500*(wr-br)+900*(wq-bq)
        
        pawnsq = sum([self.pawntable[i] for i in self.board.pieces(chess.PAWN, chess.WHITE)])
        pawnsq= pawnsq + sum([-self.pawntable[chess.square_mirror(i)] 
                                        for i in self.board.pieces(chess.PAWN, chess.BLACK)])
        knightsq = sum([self.knightstable[i] for i in self.board.pieces(chess.KNIGHT, chess.WHITE)])
        knightsq = knightsq + sum([-self.knightstable[chess.square_mirror(i)] 
                                        for i in self.board.pieces(chess.KNIGHT, chess.BLACK)])
        bishopsq= sum([self.bishopstable[i] for i in self.board.pieces(chess.BISHOP, chess.WHITE)])
        bishopsq= bishopsq + sum([-self.bishopstable[chess.square_mirror(i)] 
                                        for i in self.board.pieces(chess.BISHOP, chess.BLACK)])
        rooksq = sum([self.rookstable[i] for i in self.board.pieces(chess.ROOK, chess.WHITE)]) 
        rooksq = rooksq + sum([-self.rookstable[chess.square_mirror(i)] 
                                        for i in self.board.pieces(chess.ROOK, chess.BLACK)])
        queensq = sum([self.queenstable[i] for i in self.board.pieces(chess.QUEEN, chess.WHITE)]) 
        queensq = queensq + sum([-self.queenstable[chess.square_mirror(i)] 
                                        for i in self.board.pieces(chess.QUEEN, chess.BLACK)])
        kingsq = sum([self.kingstable[i] for i in self.board.pieces(chess.KING, chess.WHITE)]) 
        kingsq = kingsq + sum([-self.kingstable[chess.square_mirror(i)] 
                                        for i in self.board.pieces(chess.KING, chess.BLACK)])
        
        self.boardvalue = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
        
        return self.boardvalue

    def quiesce(self, alpha, beta):
        stand_pat = self.evaluate_board()
        if( stand_pat >= beta ):
            return beta
        if( alpha < stand_pat ):
            alpha = stand_pat

        for move in self.board.legal_moves:
            if self.board.is_capture(move):
                self.make_move(move)        
                score = -self.quiesce( -beta, -alpha )
                self.unmake_move()

                if( score >= beta ):
                    return beta
                if( score > alpha ):
                    alpha = score  
        return alpha

    def alphabeta(self, alpha, beta, depthleft ):
        bestscore = -9999
        if( depthleft == 0 ):
            return self.quiesce( alpha, beta )
        for move in self.board.legal_moves:
            self.make_move(move)   
            score = -self.alphabeta( -beta, -alpha, depthleft - 1 )
            self.unmake_move()
            if( score >= beta ):
                return score
            if( score > bestscore ):
                bestscore = score
            if( score > alpha ):
                alpha = score   
        return bestscore

    def selectmove(self, depth):
        bestMove = chess.Move.null()
        bestValue = -99999
        alpha = -100000
        beta = 100000
        for move in self.board.legal_moves:
            self.make_move(move)
            boardValue = -self.alphabeta(-beta, -alpha, depth-1)
            if boardValue > bestValue:
                bestValue = boardValue
                bestMove = move
            if( boardValue > alpha ):
                alpha = boardValue
            self.unmake_move()
        self.movehistory.append(bestMove)
        return bestMove
    
    def update_eval(self, mov, side):        
        #update piecequares
        movingpiece = self.board.piece_type_at(mov.from_square)
        if side:
            self.boardvalue = self.boardvalue - self.tables[movingpiece - 1][mov.from_square]
            #update castling
            if (mov.from_square == chess.E1) and (mov.to_square == chess.G1):
                self.boardvalue = self.boardvalue - self.rookstable[chess.H1]
                self.boardvalue = self.boardvalue + self.rookstable[chess.F1]
            elif (mov.from_square == chess.E1) and (mov.to_square == chess.C1):
                self.boardvalue = self.boardvalue - self.rookstable[chess.A1]
                self.boardvalue = self.boardvalue + self.rookstable[chess.D1]
        else:
            self.boardvalue = self.boardvalue + self.tables[movingpiece - 1][mov.from_square]
            #update castling
            if (mov.from_square == chess.E8) and (mov.to_square == chess.G8):
                self.boardvalue = self.boardvalue + self.rookstable[chess.H8]
                self.boardvalue = self.boardvalue - self.rookstable[chess.F8]
            elif (mov.from_square == chess.E8) and (mov.to_square == chess.C8):
                self.boardvalue = self.boardvalue + self.rookstable[chess.A8]
                self.boardvalue = self.boardvalue - self.rookstable[chess.D8]
            
        if side:
            self.boardvalue = self.boardvalue + self.tables[movingpiece - 1][mov.to_square]
        else:
            self.boardvalue = self.boardvalue - self.tables[movingpiece - 1][mov.to_square]
            
        
        #update material #Cambiado del original
        #if mov.drop != None:
        if self.board.is_capture(mov):
            if side:
                #self.boardvalue = self.boardvalue + self.piecevalues[mov.drop-1]
                if self.board.is_en_passant(mov):
                    self.boardvalue = self.boardvalue + self.piecevalues[chess.PAWN - 1] 
                else:
                    self.boardvalue = self.boardvalue + self.piecevalues[self.board.piece_at(mov.to_square).piece_type - 1] 
            else:
                #self.boardvalue = self.boardvalue - self.piecevalues[mov.drop-1]
                if self.board.is_en_passant(mov):
                    self.boardvalue = self.boardvalue - self.piecevalues[chess.PAWN - 1] 
                else:
                    self.boardvalue = self.boardvalue - self.piecevalues[self.board.piece_at(mov.to_square).piece_type - 1]

                
        #update promotion
        if mov.promotion != None:
            if side:
                self.boardvalue = self.boardvalue + self.piecevalues[mov.promotion-1] - self.piecevalues[movingpiece-1]
                self.boardvalue = self.boardvalue - self.tables[movingpiece - 1][mov.to_square] \
                    + self.tables[mov.promotion - 1][mov.to_square]
            else:
                self.boardvalue = self.boardvalue - self.piecevalues[mov.promotion-1] + self.piecevalues[movingpiece-1]
                self.boardvalue = self.boardvalue + self.tables[movingpiece - 1][mov.to_square] \
                    - self.tables[mov.promotion - 1][mov.to_square]
                
                
        return mov

    def make_move(self, mov):
        self.update_eval(mov, self.board.turn)
        self.board.push(mov)
        
        return mov

    def unmake_move(self):
        mov = self.board.pop()
        self.update_eval(mov, not self.board.turn)
        
        return mov

    def evaluate_board(self):
        if self.board.is_checkmate():
            if self.board.turn:
                return -9999
            else:
                return 9999
        if self.board.is_stalemate():
            return 0
        if self.board.is_insufficient_material():
            return 0
        
        eval = self.boardvalue
        if self.board.turn:
            return eval
        else:
            return -eval

    def predict_move(self, board):
        self.board = board.copy()
        move = self.selectmove(self.depth)

        return move
    
    def predict_state_alphabeta(self, board, move):
        board_copy = board.copy()
        alpha = -100000
        beta = 100000
        if board_copy.is_valid():
            self.board = board_copy
            self.make_move(move)
            boardValue = -self.alphabeta(-beta, -alpha, self.depth-1)
            self.unmake_move()

            value = np.float64(boardValue/9999)
            if value > 1:
                value = np.float64(0.9999)
            elif value < -1:
                value = np.float64(-0.9999)
        else:
            value = -1

        return value
    
    def predict_state(self, board, move):
        board_copy = board.copy()

        if board_copy.is_valid():
            self.board = board_copy
            self.update_eval(move, self.board.turn)
            value = self.evaluate_board()
        else:
            value = -9999
        value = np.float64(value/9999)
        if value > 1:
            value = np.float64(0.9999)
        elif value < -1:
            value = np.float64(-0.9999)
   
        return value
    
    def set_board(self, board):
        if board.is_valid():
            self.board = board
        else:
            print("Error tablero no valido")

    
"""
Autor: Arjan Groen
Url: https://github.com/arjangroen/RLC/tree/master

This player integrates the RLC models with the MCTS algorithm
"""
class MonteCarloAgent:
    def __init__(self, model_to_load, reward_type = "greedy"):
        opponent = agent_rlc.GreedyAgent()
        if reward_type == "stockfish":
            rew = "stockfish"
        elif reward_type == "minmax":
            rew = "minmax"
        else:
            rew = "greedy"
        env = environment_rlc.Board(opponent, FEN=None, reward=rew)
        env.reset()
        player = agent_rlc.Agent(lr=0.01, network='big')
        player.model.load_weights(model_to_load)    # model of format <model_name>.h5
        self.gamma = 0.8
        self.learner = learn_rlc.TD_search(env, player, gamma=self.gamma, search_time=2)
        self.learner.agent.fix_model()
    
    def montecarlo_search(self):
        tree = tree_rlc.Node(self.learner.env.board, gamma=self.gamma)  # Initialize the game tree

        tree = self.learner.mcts(tree)
        max_move = None
        max_value = np.NINF

        for move, child in tree.children.items():
            sampled_value = np.mean(child.values)

            if sampled_value > max_value:
                max_value = sampled_value
                max_move = move

        return max_move

    def mirror_move(self, move):
        return chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )

    def predict_move(self, board):
        board_copy = board.copy()
        turn = board.turn

        #MCTS works only with white players, so for the black player, the board and the predicted move are mirrored
        if turn == chess.BLACK:
            board_copy = board_copy.mirror()

        self.learner.env.board = board_copy
        move = self.montecarlo_search()

        if turn == chess.BLACK:
            move = self.mirror_move(move)
        
        return move

#This agents allows the user to play
class ManualAgent:
    def __init__(self):
        pass

    def print_legal_moves(self, board):
        legal_moves = list(board.legal_moves)
        print("\nMovimientos legales: ", end= "")
        for move in legal_moves:
            print(board.san(move), end=", ")
        print("\n")

    def predict_move(self, board):
        board_copy = board.copy()
        self.print_legal_moves(board_copy)
        bad_move = True
        while bad_move:
            move_player = input("Introduce la jugada (Ejemplo a2a4, Qd1d2, O-O-O, dxe4, Nxf6, e8Q, mirar jugadas legales, enter para aleatorio):\n")
            if move_player == "":
                move_player = random.choice(list(board_copy.legal_moves))
            else:
                try:
                    board_copy.push_san(move_player)
                except ValueError:
                    print("Invalid move!")
                    continue
                move_player = board.parse_san(move_player)
            bad_move = False
        
        return move_player
    
#This player integrates the models trained with PPO, the model used will the one named ppo_model
class PpoAgent:
    def __init__(self):
        self.player_selection = None
        self.board = chess.Board()

        try:
            self.latest_policy = "ppo_agent/ppo_model"
        except ValueError:
            print("Policy not found.")
            exit(0)

        with suppress_stdout():
            for previous_board_number in range(7, 0, -1): #Load the model with the correct observation size
                try: 
                    self.NumPreviousBoards = previous_board_number
                    self.board_history = np.zeros((8, 8, (1+self.NumPreviousBoards)*13), dtype=bool)
                    env_kwargs = {"logger": False, "evaluate": True, "previous_boards": self.NumPreviousBoards}

                    env = ppo_agent.chess_env.chess.env(**env_kwargs)

                    env = ppo_agent.one_agent_chess.SB3ActionMaskWrapper(env)

                    env.reset()

                    env = ppo_agent.one_agent_chess.ActionMasker(env, ppo_agent.one_agent_chess.mask_fn)

                    self.model = MaskablePPO.load(self.latest_policy, env = env)
                    break
                except ValueError:
                    pass


    def observe_board(self):
        current_index = not self.board.turn

        observation = chess_utils.get_observation(self.board, current_index)
        observation = np.dstack((observation[:, :, :7], self.board_history))
        # We need to swap the white 6 channels with black 6 channels
        if current_index == 1:
            # 1. Mirror the board
            observation = np.flip(observation, axis=0)
            # 2. Swap the white 6 channels with the black 6 channels
            #for i in range(1, 9):
            for i in range(1, self.NumPreviousBoards+2):
                tmp = observation[..., 13 * i - 6 : 13 * i].copy()
                observation[..., 13 * i - 6 : 13 * i] = observation[
                    ..., 13 * i : 13 * i + 6
                ]
                observation[..., 13 * i : 13 * i + 6] = tmp
        legal_moves = (
            chess_utils.legal_moves(self.board) if current_index == self.player_selection else []
        )

        action_mask = np.zeros(4672, "int8")
        for i in legal_moves:
            action_mask[i] = 1
        
        return {"observation": observation, "action_mask": action_mask}
    
    def observe_board_after_move(self):
        current_index = self.board.turn

        observation = chess_utils.get_observation(self.board, current_index)
        observation = np.dstack((observation[:, :, :7], self.board_history))
        # We need to swap the white 6 channels with black 6 channels
        if current_index == 1:
            # 1. Mirror the board
            observation = np.flip(observation, axis=0)
            # 2. Swap the white 6 channels with the black 6 channels
            #for i in range(1, 9):
            for i in range(1, self.NumPreviousBoards+2):
                tmp = observation[..., 13 * i - 6 : 13 * i].copy()
                observation[..., 13 * i - 6 : 13 * i] = observation[
                    ..., 13 * i : 13 * i + 6
                ]
                observation[..., 13 * i : 13 * i + 6] = tmp
        legal_moves = (
            chess_utils.legal_moves(self.board) if current_index == self.player_selection else []
        )

        action_mask = np.zeros(4672, "int8")
        for i in legal_moves:
            action_mask[i] = 1
        
        return {"observation": observation, "action_mask": action_mask}

    #Does not work
    def predict(self, board):
        #Function should be used to evaluate the position after a move
        self.board = board.copy()
        observed_board = self.observe_board_after_move()
        observation = observed_board["observation"]

        return self.model.policy.predict_values(self.model.policy.obs_to_tensor(observation)[0])[0]

    #Does not work
    def predict_move_by_value(self, board):
        self.player_selection = 0
        board_copy = board.copy()
        max_value = np.NINF
        max_move = None
        print("initial value", self.predict(board_copy).item())

        for move in board_copy.legal_moves:
            two_moves = False
            board_copy.push(move)

            value = self.predict(board_copy).item()
            if value > max_value:
                max_value = value
                max_move = move
            print(value, max_value, move, max_move)
            board_copy.pop()

        board_copy.push(max_move)
        self.board = board_copy
        
        next_board = chess_utils.get_observation(self.board, player=0)
        self.board_history = np.dstack(
                (next_board[:, :, 7:], self.board_history[:, :, :-13])
            )

        return max_move

    
    def predict_move(self, board):
        board_copy = board.copy()
        self.board = board_copy
        current_index = not self.board.turn

        #Store the previous move in the board history and check if the player is playing as white or black
        try:
            previous_move = board_copy.pop()
            # We always take the perspective of the white agent for the board history
            next_board = chess_utils.get_observation(board_copy, player=0)
            self.board_history = np.dstack(
                (next_board[:, :, 7:], self.board_history[:, :, :-13])
            )
            board_copy.push(previous_move)

            #The player is black because there was a previous move
            if self.player_selection is None:
                self.player_selection = 1
        except:
            #The player is white because there was not a previous move
            if self.player_selection is None:
                self.player_selection = 0

        observed_board = self.observe_board()
        observation = observed_board["observation"]
        action_mask = observed_board["action_mask"]        
        prediction = self.model.predict(
                observation, action_masks=action_mask, deterministic=True
            )

        act = int(
            prediction[0]
        )
        action = int(act)
   
        chess_utils.legal_moves(self.board)
        chosen_move = chess_utils.action_to_move(self.board, action, current_index)
        assert chosen_move in self.board.legal_moves

        next_board = chess_utils.get_observation(self.board, player=0)
        self.board_history = np.dstack(
            (next_board[:, :, 7:], self.board_history[:, :, :-13])
        )

        return chosen_move