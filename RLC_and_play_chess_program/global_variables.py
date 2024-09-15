import chess_agents

"""
Parameters: play_chess.py
"""
#All the available players, put in a list for a easier change of player 
players_to_choose = [                                  #Index
    "random",                                          #  0
    "stockfish",                                       #  1
    "stockfish6",                                      #  2
    "capture",                                         #  3
    "manual",                                          #  4
    "ppo",                                             #  5
    "min_max_2",                                       #  6
    "min_max_4",                                       #  7
    "trained_model_rlc"                                #  8
    "montecarlo",                                      #  9
    "montecarlo_greedy_5_c10",                         # 10
    "montecarlo_greedy_26",                            # 11
    "montecarlo_greedy_51",                            # 12
    "montecarlo_greedy_101",                           # 13
    "montecarlo_greedy_176",                           # 14
    "montecarlo_minmax_opponent",                      # 15
    "montecarlo_stockfish_opponent",                   # 16
    "montecarlo_greedy_op_rew_sto_after_before_5_c2",  # 17
    "montecarlo_greedy_op_rew_sto_after_before_25_c2", # 18
    "montecarlo_minmax_rew_greedy_op_5_c10_v4",        # 19
    "montecarlo_rew_stockfish_op_stockfish_51",        # 20
    "montecarlo_model_greedy_op_minmax_rew_5_c2",      # 21
    "montecarlo_model_sto_op_greedy_rew_5_c2",         # 22
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