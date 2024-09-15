import chess
import chess_agents
import random
import global_variables as gv

# Function to print the chessboard with colored letters and numbers
def print_board(board):
    color_orange = "\033[38;5;208m"
    color_end = "\033[0m"

    #os.system('cls' if os.name == 'nt' else 'clear')    
    print("\n")
    
    for i, row in enumerate(str(board).split('\n')):
        print(color_orange + f"{8 - i} " + color_end + "".join(colorize_piece(piece) for piece in row))
    print("  " + color_orange + "a b c d e f g h" + color_end)

# Function to colorize the piece representation
def colorize_piece(piece):
    color_grey = "\033[90m"
    color_end = "\033[0m"

    #If the piece is lower is black
    if piece.islower():
        return color_grey + piece + color_end
    return piece

#Function to instantiate the player
def get_player_agent(player):
    if player == "trained_model_rlc":
        agent = chess_agents.AgentRLCModel()
    elif player == "min_max_4":
        agent = chess_agents.MinMaxAlphaBetaIncrementalEvaluationAgent(depth=4)
    elif player == "min_max_2":
        agent = chess_agents.MinMaxAlphaBetaIncrementalEvaluationAgent(depth=2)
    elif player == "stockfish":
        agent = chess_agents.StockfishAgent(depth = 1)
    elif player == "stockfish6":
        agent = chess_agents.StockfishAgent(depth = 6, parameters={"Skill Level": 20, "Threads":1, "Minimum Thinking Time": 20, "UCI_LimitStrength": False, "UCI_Elo": 1350})
    elif player == "montecarlo":
        agent = chess_agents.MonteCarloAgent("RLC_model.h5")
    elif player == "manual":
        agent = chess_agents.ManualAgent()
    elif player == "random" :
        agent = chess_agents.RandomAgent()
    elif player == "montecarlo_stockfish_opponent":
        agent = chess_agents.MonteCarloAgent("RLC_model_trained_stockfish.h5")
    elif player == "capture":
        agent = chess_agents.CaptureAgent()
    elif player == "montecarlo_minmax_opponent":
        agent = chess_agents.MonteCarloAgent("RLC_model_trained_minmax.h5")
    elif player == "montecarlo_rew_stockfish":
        agent = chess_agents.MonteCarloAgent("RLC_model_stockfish_reward.h5")
    elif player == "montecarlo_rew_minmax":
        agent = chess_agents.MonteCarloAgent("RLC_model_minmax_reward.h5")
    elif player == "ppo":
        agent = chess_agents.PpoAgent()
    elif player == "montecarlo_greedy_26":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_26.h5")
    elif player == "montecarlo_greedy_51":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_51.h5")
    elif player == "montecarlo_greedy_101":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_101.h5")
    elif player == "montecarlo_greedy_176":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_176.h5")
    elif player == "montecarlo_stockfish_op_rew_5":
        agent = chess_agents.MonteCarloAgent("RLC_model_stockfish_5_iter.h5", "greedy")#"stockfish")
    elif player == "montecarlo_stockfish_minmax_rew_5":
        agent = chess_agents.MonteCarloAgent("RLC_model_minmax_reward_5_iter.h5", "greedy")#"minmax")
    elif player == "montecarlo_rew_stockfish_op_greedy_6":
        agent = chess_agents.MonteCarloAgent("RLC_model_rew_stockfish_op_greedy_6.h5")
    elif player == "montecarlo_rew_stockfish_op_stockfish_51":
        agent = chess_agents.MonteCarloAgent("RLC_model_rew_stockfish_op_stockfish_51.h5")
    elif player == "montecarlo_sto_op_rew_6_v2":
        agent = chess_agents.MonteCarloAgent("RLC_model_sto_op_rew_6_v2.h5")
    elif player == "montecarlo_minmax_rew_greedy_op_6_v2":
        agent = chess_agents.MonteCarloAgent("RLC_model_minmax_rew_greedy_op_6_v2.h5")
    elif player == "montecarlo_sto_rew_greedy_op_6_v2":
        agent = chess_agents.MonteCarloAgent("RLC_model_sto_rew_greedy_op_6_v2.h5")
    elif player == "montecarlo_minmax_rew_greedy_op_5_c10_v4":
        agent = chess_agents.MonteCarloAgent("RLC_model_minmax_rew_greedy_op_5_c10_v4.h5")#, reward_type="minmax")
    elif player == "montecarlo_greedy_101_v2":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_101_v2.h5")
    elif player == "montecarlo_model_greedy_op_rew_sto_25":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_op_rew_sto_25.h5")
    elif player == "montecarlo_greedy_5_c10":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_5_c10.h5")
    elif player == "montecarlo_greedy_op_rew_sto_after_5_c2":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_op_rew_sto_after_5_c2.h5")
    elif player == "montecarlo_greedy_op_rew_sto_after_before_5_c2":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_op_rew_sto_after_before_5_c2.h5")
    elif player == "montecarlo_model_greedy_op_rew_sto_after_25_c2":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_op_rew_sto_after_25_c2.h5")
    elif player == "montecarlo_greedy_op_rew_sto_after_before_25_c2":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_op_rew_sto_after_before_25_c2.h5")
    elif player == "montecarlo_model_greedy_op_minmax_rew_5_c2":
        agent = chess_agents.MonteCarloAgent("RLC_model_greedy_op_minmax_rew_5_c2.h5")
    elif player == "montecarlo_model_sto_op_greedy_rew_5_c2":
        agent = chess_agents.MonteCarloAgent("RLC_model_sto_op_greedy_rew_5_c2.h5")
    else:
        print("Modelo a cargar sin especificar")
        exit()

    return agent

# Main function to play chess
def play_chess(player_0, player_1, print_boards, first_move_random):
    board = chess.Board()

    if first_move_random:
        move_player_0 = random.choice(list(board.legal_moves))
        
        #Only used to prevent a bug when using PpoAgent
        if isinstance(player_0, chess_agents.PpoAgent):
            player_0.predict_move(board)
        
        board.push(move_player_0)

        move_player_1 = random.choice(list(board.legal_moves))
        
        if isinstance(player_1, chess_agents.PpoAgent):
            player_1.predict_move(board)
        
        board.push(move_player_1)

    if print_boards:
        print_board(board)
            
    while not board.is_game_over():
        if board.can_claim_draw():
            print("\nTablas reclamada")
            break

        player_0_move = player_0.predict_move(board)

        board.push(player_0_move)
        if board.is_game_over():
            break
        elif board.can_claim_draw():
            print("\nTablas reclamada")
            break
        if print_boards:
            print_board(board)

        player_1_move = player_1.predict_move(board)

        board.push(player_1_move)
        if print_boards:
            print_board(board)
            print(f"\nJugadas: {player_0_move} -- {player_1_move}")
        
    print_board(board)
    print("\nPartida finalizada")

    return board.result(claim_draw=True)

if __name__ == "__main__":
    #Index:            0                1           2            3           4         5          6        7
    # players_to_choose = ["trained_model_rlc", "min_max_4", "stockfish", "montecarlo", "random", "manual", "capture", "ppo",
    # #                     8                               9                           10                         11 
    #         "montecarlo_stockfish_opponent", "montecarlo_minmax_opponent", "montecarlo_rew_stockfish", "montecarlo_rew_minmax",
    # #            12                  13                14                 15
    #         "stockfish6", "montecarlo_greedy_176", "min_max_2", "rlc_stockfish_op_rew_5",
    # #                 16                      17                     18                         19
    #         "montecarlo_greedy_26", "montecarlo_greedy_51", "montecarlo_greedy_101", "rlc_stockfish_minmax_rew_5",
    # #                    20                               21                               22
    #         "rlc_rew_stockfish_op_greedy_6", "rlc_rew_stockfish_op_stockfish_51", "rlc_sto_op_rew_6_v2",
    # #                    23                              24                                     25
    #         "rlc_minmax_rew_greedy_op_6_v2", "rlc_sto_rew_greedy_op_6_v2", "rlc_minmax_rew_greedy_op_5_c10_v4",
    # #                  26                        27                  28                       29
    #         "montecarlo_greedy_101_v2", "rlc_op_rew_sto_25", "rlc_greedy_5_c10", "rlc_greedy_op_rew_sto_after_5_c2",
    # #                     30                                            31                                              32
    #         "rlc_greedy_op_rew_sto_after_before_5_c2", "RLC_model_greedy_op_rew_sto_after_25_c2.h5", "RLC_model_greedy_op_rew_sto_after_before_25_c2.h5",
    # #                        33                                    34
    #         "RLC_model_greedy_op_minmax_rew_5_c2", "RLC_model_sto_op_greedy_rew_5_c2"]
    
    #Parameters to change
    number_of_games = gv.number_of_games
    print_boards = gv.print_boards
    first_move_random = gv.print_boards
    #player_0 has the white pieces, while the player_1 has the black pieces
    player_0 = gv.player_0
    player_1 = gv.player_1

    p0 = get_player_agent(player_0)
    p1 = get_player_agent(player_1)

    #To keep track of the results
    white_victories = 0
    black_victories = 0
    draws = 0

    for i in range(number_of_games):
        print("partida numero", i)
        print(f"\n{player_0} como blancas contra {player_1} como negras")
        result = play_chess(p0, p1, print_boards, first_move_random)

        if result == "1-0":
            white_victories += 1
        elif result == "0-1":
            black_victories += 1
        elif result == "1/2-1/2":
            draws += 1
        else:
            assert False, "bad result"

        print(f"{player_0} victories: {white_victories}")
        print(f"{player_1} victories: {black_victories}")
        print(f"draws: {draws}")
