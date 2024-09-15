import numpy as np
import chess

#Class used to convert a board into a layer board and viceversa
class ChessBoardProcessor:
    def __init__(self):
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
        self.reverse_mapper = {v: k for k, v in self.mapper.items()}

    #Given the board, returns the equivalent layer board
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

    #Given a layer board, it returns the equivalent board
    def reconstruct_board(self, layer_board):
        """
        Reconstructs the board from its numerical representation.
        
        :param layer_board: A numpy array representing the board state
        :return: A chess.Board object
        """
        board = chess.Board()  # Initialize an empty board
        board.clear()  # Clear the board
        
        layer_board1 = layer_board[0, :, :, :]

        # Reconstruct the pieces from layers 0-5
        for layer in range(6):
            for row in range(8):
                for col in range(8):
                    sign = layer_board1[layer, row, col]
                    #print(layer_board1)
                    #print(sign)
                    if sign == 0:
                        continue
                    piece_symbol = self.reverse_mapper[layer]
                    if sign == -1:
                        piece_symbol = piece_symbol.lower()
                    elif sign == 1:
                        piece_symbol = piece_symbol.upper()

                    board.set_piece_at(row * 8 + col, chess.Piece.from_symbol(piece_symbol))
        
        # Reconstruct additional game state information from layer 6
        move_number = int(1 / np.max(layer_board1[6, :, :]))
        board.fullmove_number = move_number
        
        if layer_board1[6, 0, 0] == 1:
            board.turn = True  # White to move
        else:
            board.turn = False  # Black to move
        
        board.clear_stack()

        #There will be no castling rights
        #board.castling_rights = chess.BB_ALL

        return board

#Test for the functions
if __name__ == "__main__":
    processor = ChessBoardProcessor()

    # Create a chess board with the starting position
    board = chess.Board()

    # Convert the board to the layer representation
    layer_board = processor.get_layer_board(board)
    print("Layer Board Representation:\n", layer_board)

    # Reconstruct the board from the layer representation
    reconstructed_board = processor.reconstruct_board(np.expand_dims(layer_board, axis=0))
    print(f"Reconstructed Board:\n{reconstructed_board}")

    # Verify the reconstruction by printing the FEN notation
    print("Original FEN:     ", board.fen())
    print("Reconstructed FEN:", reconstructed_board.fen())