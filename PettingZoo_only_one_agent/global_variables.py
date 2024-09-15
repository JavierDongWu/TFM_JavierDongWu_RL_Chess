"""
Parameters: black_agent_chess.py
"""
only_evaluate_model = False       # If True, only evaluate the last trained model
only_see_model_play = False       # If True, only visualize the evaluation of the last trained model. Has lower priority than only_evaluate_model
num_step_to_train = 12_000_000    # The number of steps in the environment needed to complete the training
use_saved_model = False           # Whether to use a saved model (the last one) or not
use_logger = False                # If False, logs at INFO level; if True, logs at ERROR level
training_seed = 0                 # The seed number used to randomize the training
NumPreviousBoards = 3             # The number of stored boards in the observation; default value was 7

learning_rate=0.0015              # The learning rate of the training algorithm
ent_coef = 0.01                   # The entropy coefficient of the training algorithm
partial_steps = 200_000           # Number of steps to update the ppo model used as the opponent

tensorboard_log = "./log/MASKPPO" # The directory to save TensorBoard log data