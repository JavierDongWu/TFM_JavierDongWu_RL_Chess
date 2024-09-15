import chess_env  
import black_agent_chess
from chess_env import chess_utils
import numpy as np
from sb3_contrib import MaskablePPO
import chess


class PpoAgent:
    def __init__(self):
        self.player_selection = None
        self.NumPreviousBoards = 3
        self.board = chess.Board()
        self.board_history = np.zeros((8, 8, (1+self.NumPreviousBoards)*13), dtype=bool)
        env_kwargs = {"logger": False, "evaluate": True}

        try:
            self.latest_policy = "ppo_agent/good_against_capture_stockfish_model_better"
        except ValueError:
            print("Policy not found.")
            exit(0)

        env = chess_env.chess.env(**env_kwargs)

        env = black_agent_chess.SB3ActionMaskWrapper(env)

        env.reset()  # Must call reset() in order to re-define the spaces

        env = black_agent_chess.ActionMasker(env, black_agent_chess.mask_fn)

        self.model = MaskablePPO.load(self.latest_policy, env = env)

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

    def predict_move(self, board):
        board_copy = board.copy()
        self.board = board_copy
        current_index = not self.board.turn

        try:
            previous_move = board_copy.pop()
            # Update board after applying action
            # We always take the perspective of the white agent
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

        act = int(
            self.model.predict(
                observation, action_masks=action_mask, deterministic=False
            )[0]
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