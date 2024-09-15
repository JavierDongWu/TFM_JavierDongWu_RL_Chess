# noqa: D212, D415
"""
# Chess

```{figure} classic_chess.gif
:width: 140px
:name: chess
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic import chess_v6` |
|--------------------|------------------------------------|
| Actions            | Discrete                           |
| Parallel API       | Yes                                |
| Manual Control     | No                                 |
| Agents             | `agents= ['player_0', 'player_1']` |
| Agents             | 2                                  |
| Action Shape       | Discrete(4672)                     |
| Action Values      | Discrete(4672)                     |
| Observation Shape  | (8,8,111)                          |
| Observation Values | [0,1]                              |


Chess is one of the oldest studied games in AI. Our implementation of the observation and action spaces for chess are what the AlphaZero method uses, with two small changes.

### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

Like AlphaZero, the main observation space is an 8x8 image representing the board. It has 111 channels representing:

* Channels 0 - 3: Castling rights:
  * Channel 0: All ones if white can castle queenside
  * Channel 1: All ones if white can castle kingside
  * Channel 2: All ones if black can castle queenside
  * Channel 3: All ones if black can castle kingside
* Channel 4: Is black or white
* Channel 5: A move clock counting up to the 50 move rule. Represented by a single channel where the *n* th element in the flattened channel is set if there has been *n* moves
* Channel 6: All ones to help neural networks find board edges in padded convolutions
* Channel 7 - 18: One channel for each piece type and player color combination. For example, there is a specific channel that represents black knights. An index of this channel is set to 1 if a black knight is in the corresponding spot on the game board, otherwise, it is set to 0.
Similar to LeelaChessZero, en passant possibilities are represented by displaying the vulnerable pawn on the 8th row instead of the 5th.
* Channel 19: represents whether a position has been seen before (whether a position is a 2-fold repetition)
* Channel 20 - 111 represents the previous 7 boards, with each board represented by 13 channels. The latest board occupies the first 13 channels, followed by the second latest board, and so on. These 13 channels correspond to channels 7 - 20.


Similar to AlphaZero, our observation space follows a stacking approach, where it accumulates the previous 8 board observations.

Unlike AlphaZero, where the board orientation may vary, in our system, the `env.board_history` always maintains the orientation towards the white agent, with the white agent's king consistently positioned on the 1st row. In simpler terms, both players are observing the same board layout.

Nevertheless, we have incorporated a convenient feature, the env.observe('player_1') function, specifically for the black agent's orientation. This facilitates the training of agents capable of playing proficiently as both black and white.

#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.

### Action Space

From the AlphaZero chess paper:

> [In AlphaChessZero, the] action space is a 8x8x73 dimensional array.
Each of the 8×8 positions identifies the square from which to “pick up” a piece. The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be
moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The
next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible
underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or
rook respectively. Other pawn moves or captures from the seventh rank are promoted to a
queen.

We instead flatten this into 8×8×73 = 4672 discrete action space.

You can get back the original (x,y,c) coordinates from the integer action `a` with the following expression: `(a // (8*73), (a // 73) % 8, a % (8*73) % 73)`

Example:
    >>> x = 6
    >>> y = 0
    >>> c = 12
    >>> a = x*(8*73) + y*73 + c
    >>> print(a // (8*73), a % (8*73) // 73, a % (8*73) % 73)
    6 0 12

Note: the coordinates (6, 0, 12) correspond to column 6, row 0, plane 12. In chess notation, this would signify square G1:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| A | B | C | D | E | F | G | H |

### Rewards

| Winner | Loser | Draw |
| :----: | :---: | :---: |
| +1     | -1    | 0 |

### Version History

* v6: Fixed wrong player starting first, check for insufficient material/50-turn rule/three fold repetition (1.23.2)
* v5: Changed python-chess version to version 1.7 (1.13.1)
* v4: Changed observation space to proper AlphaZero style frame stacking (1.11.0)
* v3: Fixed bug in arbitrary calls to observe() (1.8.0)
* v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
* v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v0: Initial versions release (1.0.0)

"""
from __future__ import annotations

from os import path

import chess
import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
#from pettingzoo.classic.chess import chess_utils
from chess_env import chess_utils #Using the local chess_utils
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from logging import getLogger, basicConfig, ERROR, INFO
import random
 
def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "chess_v6",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode: str | None = None, screen_height: int | None = 800, logger = False, evaluate = False, num_previous_boards = 3):
        EzPickle.__init__(self, render_mode, screen_height)
        super().__init__()
        
        #Defines the number of previous boards saved which affects the size of the observation
        self.NumPreviousBoards = num_previous_boards

        self.evaluate = evaluate
        self.logger = getLogger()
        if logger:
            basicConfig(level=INFO, format='%(message)s')
        else:
            basicConfig(level=ERROR, format='%(message)s')

        self.board = chess.Board()

        self.agents = [f"player_{i}" for i in range(2)]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {name: spaces.Discrete(8 * 8 * 73) for name in self.agents}
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        #low=0, high=1, shape=(8, 8, 111), dtype=bool
                        low=0, high=1, shape=(8, 8, 20+self.NumPreviousBoards*13), dtype=bool 
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(4672,), dtype=np.int8
                    ),
                }
            )
            for name in self.agents
        }

        self.rewards = None
        self._cumulative_rewards = None
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}

        self.agent_selection = None

        #self.board_history = np.zeros((8, 8, 104), dtype=bool)
        self.board_history = np.zeros((8, 8, (1+self.NumPreviousBoards)*13), dtype=bool)

        self.boards_value = None
        self.FinalResult = None
        # self.king_legal_moves = None
        self.previous_mobility_score = None
        # self.steps_counter = None
        self.pieces_value = {
            'p': -1,  # Black pawn
            'n': -3,  # Black knight
            'b': -3,  # Black bishop
            'r': -5,  # Black rook
            'q': -9,  # Black queen
            'k': -50, # Black king
            'P': 1,   # White pawn
            'N': 3,   # White knight
            'B': 3,   # White bishop
            'R': 5,   # White rook
            'Q': 9,   # White queen
            'K': 50,  # White king
        }

        self.max_moves = {
            'p': 2,  # Black pawn, theoretically 4
            'n': 8,  # Black knight
            'b': 13,  # Black bishop
            'r': 14,  # Black rook
            'q': 27,  # Black queen
            'k': 8,  # Black king
            'P': 2,  # White pawn, theoretically 4
            'N': 8,  # White knight
            'B': 13,  # White bishop
            'R': 14,  # White rook
            'Q': 27,  # White queen
            'K': 8   # White king
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen_height = self.screen_width = screen_height

        self.screen = None

        if self.render_mode in ["human", "rgb_array"]:
            self.BOARD_SIZE = (self.screen_width, self.screen_height)
            self.clock = pygame.time.Clock()
            self.cell_size = (self.BOARD_SIZE[0] / 8, self.BOARD_SIZE[1] / 8)

            bg_name = path.join(path.dirname(__file__), "img/chessboard.png")
            self.bg_image = pygame.transform.scale(
                pygame.image.load(bg_name), self.BOARD_SIZE
            )

            def load_piece(file_name):
                img_path = path.join(path.dirname(__file__), f"img/{file_name}.png")
                return pygame.transform.scale(
                    pygame.image.load(img_path), self.cell_size
                )

            self.piece_images = {
                "pawn": [load_piece("pawn_black"), load_piece("pawn_white")],
                "knight": [load_piece("knight_black"), load_piece("knight_white")],
                "bishop": [load_piece("bishop_black"), load_piece("bishop_white")],
                "rook": [load_piece("rook_black"), load_piece("rook_white")],
                "queen": [load_piece("queen_black"), load_piece("queen_white")],
                "king": [load_piece("king_black"), load_piece("king_white")],
            }


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        current_index = self.possible_agents.index(agent)

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
            chess_utils.legal_moves(self.board) if agent == self.agent_selection else []
        )

        action_mask = np.zeros(4672, "int8")
        for i in legal_moves:
            action_mask[i] = 1
        
        return {"observation": observation, "action_mask": action_mask}

    def reset_orig(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        self.board = chess.Board()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        #self.board_history = np.zeros((8, 8, 104), dtype=bool)
        self.board_history = np.zeros((8, 8, (1+self.NumPreviousBoards)*13), dtype=bool)

        self.boards_value = {name: 0 for name in self.agents}
        #FinalResult puede ser 0, 1 o -1
        self.FinalResult = 2
        # self.king_legal_moves = {name: 0 for name in self.agents}
        self.previous_mobility_score = 20
        self.steps_counter = 0

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        self.board = chess.Board()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        #self.board_history = np.zeros((8, 8, 104), dtype=bool)
        self.board_history = np.zeros((8, 8, (1+self.NumPreviousBoards)*13), dtype=bool)

        self.boards_value = {name: 0 for name in self.agents}
        #FinalResult puede ser 0, 1 o -1
        self.FinalResult = 2
        # self.king_legal_moves = {name: 0 for name in self.agents}
        self.previous_mobility_score = 20
        # self.steps_counter = 0

        if self.render_mode == "human":
            self.render()

        #random.seed(random.randint(0, 10000))
        #self.action_space(self.possible_agents[0]).seed(random.randint(0, 1000000000))
        if self.evaluate:
            random_number = 0
        else:
            random_number = random.randint(0, 4)
        
        # To setup a random endgame position
        # random_position = random.randint(0, 7)
        # if random_position == 0:
        #     self.board = self._setup_endgame(chess.QUEEN, chess.WHITE)
        # elif random_position == 1:
        #     self.board = self._setup_endgame(chess.QUEEN, chess.BLACK)
        # elif random_position == 2:
        #     self.board = self._setup_endgame(chess.ROOK, chess.WHITE)
        # elif random_position == 3:
        #     self.board = self._setup_endgame(chess.ROOK, chess.BLACK)
        # elif random_position == 4:
        #     self.board = self._setup_endgame(chess.QUEEN, chess.BLACK, easy = True)
        # elif random_position == 5:
        #     self.board = self._setup_endgame(chess.QUEEN, chess.WHITE, easy = True)
        # elif random_position == 6:
        #     self.board = self._setup_endgame(chess.ROOK, chess.BLACK, easy = True)
        # elif random_position == 7:
        #     self.board = self._setup_endgame(chess.ROOK, chess.WHITE, easy = True)

        for i in range(random_number):
            obs = self.observe('player_0')                  
            observation, action_mask = obs.values()
            act = self.action_space('player_0').sample(action_mask)
            self.step(act)
            if self.terminations['player_0'] or self.terminations['player_1']:
                self.board = chess.Board()
                self.reset_orig()
                break

            obs = self.observe('player_1')                  
            observation, action_mask = obs.values()
            act = self.action_space('player_1').sample(action_mask)
            self.step(act)   
            if self.terminations['player_0'] or self.terminations['player_1']:
                self.board = chess.Board()
                self.reset_orig()
                break

    def set_game_result(self, result_val):
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            #coef_number = 3 * max(1, (160 - self.board.fullmove_number) / 32)
            coef_number = 1
            result_coef = coef_number if i == 0 else -coef_number
            self.rewards[name] += round(result_val * result_coef,7)
            self.FinalResult = result_val
            self.infos[name] = {"legal_moves": []}

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        
        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)

        #self.steps_counter += 1

        # Cast action into int
        action = int(action)

        chosen_move = chess_utils.action_to_move(self.board, action, current_index)
        assert chosen_move in self.board.legal_moves
        
        self.logger.info("before")
        self.logger.info(chess.Board(self.board.fen()))
        self.logger.info(chosen_move.uci)
        
        # previous_king_legal_moves, square_attacked = self._count_king_moves_not_in_check()
        # previous_board_is_check = self.board.is_check()
        move_is_capture = self.board.is_capture(chosen_move)
        piece_captured = self.board.piece_at(chosen_move.to_square)
        move_is_en_passant = self.board.is_en_passant(chosen_move)
        move_is_promotion = chosen_move.promotion is not None
        # previous_legal_moves = len(list(self.board.legal_moves))

        #Play the chosen move on the board
        self.board.push(chosen_move)

        self.logger.info("after")
        self.logger.info(chess.Board(self.board.fen()))

        next_legal_moves = chess_utils.legal_moves(self.board)     
        is_stale_or_checkmate = not any(next_legal_moves)   
        # claim draw is set to be true to align with normal tournament rules
        is_insufficient_material = self.board.is_insufficient_material()
        can_claim_draw = self.board.can_claim_draw()
        game_over = can_claim_draw or is_stale_or_checkmate or is_insufficient_material

        #Make that a match can only finish after a certain number of steps
        # if self.evaluate:
        #     can_claim_draw = self.board.can_claim_draw()
        #     game_over = can_claim_draw or is_stale_or_checkmate or is_insufficient_material
        # else:
        #     if self.steps_counter > 500:
        #         can_claim_draw = self.board.can_claim_draw()
        #         game_over = can_claim_draw or is_stale_or_checkmate or is_insufficient_material
        #     else:
        #         game_over = is_stale_or_checkmate or is_insufficient_material                     
        
        # Cleaning previous rewards and accumulated rewards         
        for name in self.agents:
            self.rewards[name] = 0
        if current_index == 0:
            self._cumulative_rewards['player_1'] = 0.0
        else:
            self._cumulative_rewards['player_0'] = 0.0

        # Added rewards for material advantage
        # The difference is calculated from the perspective of the white player, in the black player case, the value is negated
        next_board_value = 0
        next_board_max_mobility = 0
        next_board_white_value = 0
        pieces = self.board.piece_map()
        for key in pieces:
            piece_letter = str(pieces[key])
            piece_value = self.pieces_value[piece_letter]
            next_board_value += piece_value
            if piece_value > 0:
                next_board_white_value += piece_value
            if current_index == 0 and piece_letter.isupper():
                next_board_max_mobility += self.max_moves[str(pieces[key])]
            elif current_index == 1 and piece_letter.islower():
                next_board_max_mobility += self.max_moves[str(pieces[key])]
        
        next_board_black_value = abs(next_board_value - next_board_white_value)

        #If the value of the board for white or black is greater than 20, excluding the king(50), we say that we are in a late-game position        
        if next_board_white_value < 70 or next_board_black_value < 70:
            is_end_game = True
        else:
            is_end_game = False
            
        #Si es el turno del jugador 1, negras, se invierte el valor
        if current_index == 1:
            next_board_value = -next_board_value

        if not game_over:
            #self.logger.info(f"end game: {is_end_game}")
            #Compare the difference of material between the actual and previous board value
            board_value_change = next_board_value - self.boards_value[current_agent]
            self.logger.info(f"board value change {board_value_change}")
            reward_given = round(0.05 * board_value_change, 7)
            
            #If there is a change in board material balance, saved it and assign the reward
            if board_value_change != 0:
                if current_index == 0:
                    self.rewards[self.agents[0]] += reward_given
                    #self.rewards[self.agents[1]] -= reward_given
                    self.boards_value[self.agents[0]] = next_board_value
                    self.boards_value[self.agents[1]] = -next_board_value
                else:
                    self.rewards[self.agents[1]] += reward_given
                    #self.rewards[self.agents[0]] -= reward_given
                    self.boards_value[self.agents[0]] = -next_board_value
                    self.boards_value[self.agents[1]] = next_board_value
            
            self.logger.info(f"{next_board_value} {current_agent}")
            self.logger.info(f"Capturado {reward_given}")

            #We give rewards for improving or worsening the movility
            copied_board = self.board.copy()
            copied_board.turn = not copied_board.turn
            mobility_score = len(list(copied_board.legal_moves)) / next_board_max_mobility
            if mobility_score > self.previous_mobility_score and mobility_score <= 1 and not is_end_game:
                self.rewards[current_agent] += round(0.005 * (mobility_score - self.previous_mobility_score), 7)
            self.previous_mobility_score = mobility_score

            # We give rewards for keeping the king safe
            # current_king_legal_moves, square_attacked = self._count_king_moves_not_in_check()

            # if (current_king_legal_moves >= 2) and (previous_king_legal_moves < 2) and square_attacked and not previous_board_is_check and is_end_game:
            #     #self.rewards[current_agent] += round(0.025, 7)
            #     pass

            # We give negative rewards if the move is not safe, meaning the square has more attackers than defenders
            moving_piece = self.board.piece_at(chosen_move.to_square)

            if moving_piece.piece_type != chess.KING:
                if current_index == 0:
                    defenders_color = chess.WHITE
                    attackers_color = chess.BLACK
                else:
                    defenders_color = chess.BLACK
                    attackers_color = chess.WHITE

                dest_square = chosen_move.to_square

                attackers = list(self.board.attackers(attackers_color, dest_square))
                defenders = list(self.board.attackers(defenders_color, dest_square))

                min_value_attacker = 100 # 100 so that the king (50) can be counted as an attacker
                num_attackers = 0
                for attacker_square in attackers:
                    if not self.board.is_pinned(attackers_color, attacker_square):
                        num_attackers += 1
                        attacker_piece = self.board.piece_at(attacker_square)
                        attacker_value = abs(self.pieces_value[attacker_piece.symbol()])
                        if attacker_value < min_value_attacker:
                            min_value_attacker = attacker_value

                min_value_defender = 100
                num_defenders = 0
                for defender_square in defenders:
                    if not self.board.is_pinned(defenders_color, defender_square):
                        num_defenders += 1
                        defender_piece = self.board.piece_at(defender_square)
                        defender_value = abs(self.pieces_value[defender_piece.symbol()])
                        if defender_value < min_value_defender:
                            min_value_defender = defender_value

                moving_piece_value = abs(self.pieces_value[moving_piece.symbol()])
                
                attacker_number_advantage = num_attackers > num_defenders
                attacker_less_value = moving_piece_value > min_value_attacker

                if ((attacker_number_advantage and min_value_defender > min_value_attacker) or attacker_less_value) and not move_is_capture:
                    if attacker_less_value and not attacker_number_advantage:
                        evaluate_move_danger = moving_piece_value - min_value_attacker
                    else:
                        evaluate_move_danger = moving_piece_value
                    
                    self.rewards[current_agent] -= round(0.1*evaluate_move_danger, 7) #Double of the value for obtaining material advantage
                    self.logger.info(f"evaluation move danger: {evaluate_move_danger}")
            
                #We give negative rewards if the capture is not favored
                if move_is_capture and not move_is_en_passant and not move_is_promotion:
                    captured_piece_value = abs(self.pieces_value[piece_captured.symbol()])
                    capture_value = captured_piece_value - moving_piece_value
                    attacker_trade_pieces_value = captured_piece_value + min_value_attacker
                    if attacker_number_advantage and ((moving_piece_value + min_value_defender) > attacker_trade_pieces_value or num_defenders == 0):
                        bad_capture = True
                    elif not attacker_number_advantage and moving_piece_value > attacker_trade_pieces_value:
                        bad_capture = True
                    else:
                        bad_capture = False

                    if capture_value < 0 and num_attackers != 0 and bad_capture:
                        self.rewards[current_agent] -= round(0.1*(moving_piece_value), 7)
                        self.logger.info(f"moving piece value: {moving_piece_value}")
                    
        if game_over:
            result = self.board.result(claim_draw=True)
            result_val = chess_utils.result_to_int(result)
            self.set_game_result(result_val)

        self.logger.info("Before Rewards:")
        self.logger.info(self.rewards)

        self._accumulate_rewards()

        #Round decimals to avoid errors
        for name in self.agents:
            self._cumulative_rewards[name] = round(self._cumulative_rewards[name], 7)

        self.logger.info("After Acc Rewards:")
        self.logger.info(self._cumulative_rewards)

        # Update board after applying action
        # We always take the perspective of the white agent
        next_board = chess_utils.get_observation(self.board, player=0)
        self.board_history = np.dstack(
            (next_board[:, :, 7:], self.board_history[:, :, :-13])
        )

        self.agent_selection = (
            self._agent_selector.next()
        )

        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            return str(self.board)
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_gui()
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )

    def _render_gui(self):
        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.set_caption("Chess")
                self.screen = pygame.display.set_mode(self.BOARD_SIZE)
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface(self.BOARD_SIZE)

        self.screen.blit(self.bg_image, (0, 0))
        for square, piece in self.board.piece_map().items():
            pos_x = square % 8 * self.cell_size[0]
            pos_y = (
                self.BOARD_SIZE[1] - (square // 8 + 1) * self.cell_size[1]
            )  # offset because pygame display is flipped
            piece_name = chess.piece_name(piece.piece_type)
            piece_img = self.piece_images[piece_name][piece.color]
            self.screen.blit(piece_img, (pos_x, pos_y))

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    #Count the legal king moves and returns the number of moves and if there is a square to move that would be in check
    def _count_king_moves_not_in_check(self):
        moves_count = 0
        square_attacked = False
        
        if self.agent_selection == self.possible_agents[0]:
            color = chess.WHITE
            opposing_color = chess.BLACK
            king_square = self.board.king(color)
        else:
            color = chess.BLACK
            opposing_color = chess.WHITE
            king_square = self.board.king(color)
            
        for file_diff in range(-1, 2):
            for rank_diff in range(-1, 2):
                if file_diff == 0 and rank_diff == 0:
                    continue # Skip the current square
                new_square = king_square + file_diff + 8 * rank_diff
                if 0 <= new_square < 64:
                    piece_at_new_square = self.board.piece_at(new_square)
                    if (piece_at_new_square is None or piece_at_new_square.color != color):
                        if not self.board.is_attacked_by(opposing_color, new_square):
                            moves_count += 1
                        else:
                            square_attacked = True

        return moves_count, square_attacked
    
    # Create a endgame position with the kings and only one rook or queen of the given color
    def _setup_endgame(self, piece, color, easy = False):
        board = chess.Board()
        board.clear()

        if easy:
            if color == chess.WHITE:
                board.set_piece_at(chess.F6, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
                position = chess.D6
            else:
                board.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(chess.F3, chess.Piece(chess.KING, chess.BLACK))
                position = chess.D3
        else:
            board.set_piece_at(chess.G2, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.G7, chess.Piece(chess.KING, chess.BLACK))
            if color == chess.WHITE:
                position = chess.D1
                board.set_piece_at(chess.E1, chess.Piece(piece, color))
            else:
                position = chess.D8
                board.set_piece_at(chess.E8, chess.Piece(piece, color))
            
        board.set_piece_at(position, chess.Piece(piece, color))

        return board

