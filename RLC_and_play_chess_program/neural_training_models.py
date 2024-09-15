from RLC_modified.RLC.real_chess import agent, environment, learn, tree
from chess.pgn import Game
import global_variables as gv

#opponent = agent.GreedyAgent()
opponent = gv.opponent
env = environment.Board(opponent, FEN=None, reward=gv.reward) #reward can be either greedy, minmax or stockfish
player = agent.Agent(lr=0.0005, network='big')
learner = learn.TD_search(env, player, gamma=0.9, search_time=0.9)
node = tree.Node(learner.env.board, gamma=learner.gamma)
player.model.summary()

w_before = learner.agent.model.get_weights()

print(gv.iterations, gv.c)
def test_train():
    learner.learn(iters=gv.iterations, timelimit_seconds=gv.timelimit_seconds , c=gv.c)


test_train()

w_after = learner.agent.model.get_weights()

print("done")

learner.env.reset()
learner.search_time = 60
learner.temperature = 1/3

n_iters = 10000

learner.play_game(n_iters,maxiter=128)

pgn = Game.from_board(learner.env.board)
pgn_name = "rlc_pgn_" + gv.name
with open(pgn_name,"w") as log:
    log.write(str(pgn))

model_name = "RLC_" + gv.name + ".h5"
learner.agent.model.save(model_name)
print("Model saved")
print(model_name)
