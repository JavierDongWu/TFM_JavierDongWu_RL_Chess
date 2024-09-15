"""
Parameters: black_agent_chess.py
"""
only_evaluate_model = False
only_see_model_play = False # Has a lower priority than only_evaluate_model
num_step_to_train = 12_000_000
use_saved_model = False
use_logger = False #INFO or ERROR level
training_seed = 0
NumPreviousBoards = 3

learning_rate=0.0015
ent_coef = 0.01
partial_steps = 200_000 # Number of steps to update the ppo model used as opponent

tensorboard_log = "./log/MASKPPO"