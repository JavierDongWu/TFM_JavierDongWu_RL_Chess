import chess_agents

"""
Parameters: play_chess.py
"""
#All the available players, put in a list for a easier change of player 
players_to_choose = [                                  #Index  #Description
    "random",                                          #  0     Player that plays randomly
    "stockfish",                                       #  1     Stockfish with depth 1
    "stockfish6",                                      #  2     Stockfish with depth 6
    "capture",                                         #  3     Player that always captures if possible, if not then it moves randomly
    "manual",                                          #  4     Play manually
    "ppo",                                             #  5     The player using a PPO model
    "min_max_2",                                       #  6     Chess program with Minmax algorithm at depth 2
    "min_max_4",                                       #  7     Chess program with Minmax algorithm at depth 4
    "trained_model_rlc"                                #  8     The model of the RLC repository provided by the author
    "montecarlo",                                      #  9     The model of the RLC repository provided by the author using MCTS 
    "montecarlo_greedy_5_c10",                         # 10     RLC MCTS model trained in the same conditions as the one provided by the author
    "montecarlo_greedy_26",                            # 11     Same as the previous one but trained for 26 iterations
    "montecarlo_greedy_51",                            # 12     Same as the previous one but trained for 51 iterations
    "montecarlo_greedy_101",                           # 13     Same as the previous one but trained for 101 iterations
    "montecarlo_greedy_176",                           # 14     Same as the previous one but trained for 176 iterations
    "montecarlo_minmax_opponent",                      # 15     RLC MCTS model 5 iterations with the Minmax program as the opponent
    "montecarlo_stockfish_opponent",                   # 16     RLC MCTS model 5 iterations with the Stockfish engine as the opponent
    "montecarlo_greedy_op_rew_sto_after_before_5_c2",  # 17     RLC MCTS model 5 iterations with rewards based on Stockfish
    "montecarlo_greedy_op_rew_sto_after_before_25_c2", # 18     RLC MCTS model 25 iterations with rewards based on Stockfish
    "montecarlo_minmax_rew_greedy_op_5_c10_v4",        # 19     RLC MCTS model 5 iterations with rewards based on the Minmax program
    "montecarlo_rew_stockfish_op_stockfish_51",        # 20     RLC MCTS model 51 iterations using Stockfish as opponent and for the rewards
    "montecarlo_model_greedy_op_minmax_rew_5_c2",      # 21     RLC MCTS model 5 iterations, c=2 with rewards based on the Minmax program
    "montecarlo_model_sto_op_greedy_rew_5_c2",         # 22     RLC MCTS model 5 iterations, c=2 with Stockfish as the opponent
]
    
number_of_games = 100     #Total number of games played
print_boards = False      #If you want every move to be printed in the console
first_move_random = False #If you want the randomize the start of the matches

#Choose a player from the list using the index number, player_0 has the white pieces, while the player_1 has the black pieces
player_0 = players_to_choose[5]
player_1 = players_to_choose[0]



"""
Parameters: neural_training_models.py
"""
#The opponent the agent will face in the training, it can be GreedyAgent, StockfishAgent or MinMaxAlphaBetaIncrementalEvaluationAgent
opponent = chess_agents.GreedyAgent()
#opponent = chess_agents.StockfishAgent(depth=6, parameters={"Minimum Thinking Time": 20, "UCI_LimitStrength": False, "UCI_Elo": 1350})
#opponent = chess_agents.MinMaxAlphaBetaIncrementalEvaluationAgent(1)

#The type of reward given, it can be "greedy", "minmax" or "stockfish"
reward = "greedy"

iterations = 5             #The number of iterations, games played needed to finish the training
timelimit_seconds = 360000 #The maximum time the training can last, in seconds
c = 2                      #For the neural network update frecuency, in number of iterations

name = "name" #The name of the trained model