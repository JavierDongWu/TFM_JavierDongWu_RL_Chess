"""Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Author: Elliot (https://github.com/elliottower)
Modified by: Javier Dong Wu
"""
import glob
import os
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils

from chess_env import chess
import random
import datetime
import global_variables as gv

from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        #super().step(action)

        # The reward, termination, truncation and info is obtained from the agent making the action
        current_agent = self.agent_selection 
        super().step(action) 
        reward = self._cumulative_rewards[current_agent]
  
        termination = self.terminations[current_agent]
        truncation = self.truncations[current_agent]
        info = self.infos[current_agent]

        new_agent = self.agent_selection

        new_observation = self.observe(new_agent)       
        
        return new_observation, reward, termination, truncation, info
        #return super().last()

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


def train_action_mask(env_fn, steps=10_000, seed=0, saved_model = None, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    partial_steps = gv.partial_steps
    opponent_updates = steps//partial_steps
    
    for opponent_update in range(opponent_updates):
        print(opponent_update, "de", opponent_updates)
        try:
            opponent_policy = "training_opponent_model"
            env_kwargs["opponent_model"] = MaskablePPO.load(opponent_policy)
        except (ValueError, FileNotFoundError):
            print("Policy does not exist yet, playing against random until there is one")
            env_kwargs["opponent_model"] = None
        
        env = env_fn.env(**env_kwargs)

        print(f"Starting training on {str(env.metadata['name'])}.")

        # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
        env = SB3ActionMaskWrapper(env)

        seed += 1
        env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

        env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
        # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
        # with ActionMasker. If the wrapper is detected, the masks are automatically
        # retrieved and used when learning. Note that MaskablePPO does not accept
        # a new action_mask_fn kwarg, as it did in an earlier draft.
        if opponent_update == 0:
            if saved_model is None:    
                model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, learning_rate=gv.learning_rate, ent_coef = gv.ent_coef, tensorboard_log="./log/MASKPPO/Opponent_update_training/black")
                #model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, learning_rate=0.003, ent_coef = 0.01)
            else:
                model = saved_model
                model.set_env(env)
        else:
            model.set_env(env)

        model.set_random_seed(seed)
        
        model.learn(total_timesteps=partial_steps, progress_bar=True)

        model.save("training_opponent_model.zip")

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[0]}."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)

    winner0=0
    winner1=0
    Nowinner=0

    for i in range(num_games):
        print("Partida numero", i)
        env.reset(seed = random.randint(0, 1000000000))

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                #Count the result of the matches
                if env.FinalResult == 1:
                    winner0 += 1
                elif env.FinalResult == -1:
                    winner1 += 1
                elif env.FinalResult == 0:
                    Nowinner += 1
                else:
                    print("Resultado final no esperado")

                break
            else:
                #if agent == env.possible_agents[0]:
                act = int(
                    model.predict(
                        observation, action_masks=action_mask, deterministic=True
                    )[0]
                )
                #else:
                #    act = env.action_space(agent).sample(action_mask)

            env.step(act)
    env.close()

    print("Winner 0: ", winner0*100/num_games, "%")
    print("Winner 1: ", winner1*100/num_games, "%")
    print("Nowinner: ", Nowinner*100/num_games, "%")


if __name__ == "__main__":
    #Measure the time it took to train
    start_time = datetime.datetime.now()

    env_fn = chess

    #Parámetros por comodidad, ordenados por prioridad
    only_evaluate_model = gv.only_evaluate_model
    only_see_model_play = gv.only_see_model_play
    num_step_to_train = gv.num_step_to_train
    use_saved_model = gv.use_saved_model
    use_logger = gv.use_logger #INFO or ERROR level
    training_seed = gv.training_seed

    env_kwargs = {"logger": use_logger, "evaluate": only_evaluate_model, "num_previous_boards": gv.NumPreviousBoards}

    #Added functionality to train save models
    if not only_see_model_play and not only_evaluate_model:
        env = env_fn.env(render_mode=None, **env_kwargs)
        if(use_saved_model):
                latest_policy = max(
                    glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
                )
                model = MaskablePPO.load(latest_policy)
                print("Usando modelo guardado", latest_policy)
                train_action_mask(env_fn, steps=num_step_to_train, seed=training_seed,saved_model = model, **env_kwargs )
        else:
            train_action_mask(env_fn, steps=num_step_to_train, seed=training_seed, saved_model = None, **env_kwargs)

        #Evaluate model after training
        env_kwargs = {"logger": use_logger, "evaluate": True, "num_previous_boards": gv.NumPreviousBoards}
        eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs)
    elif only_evaluate_model:
        eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs)
    elif only_see_model_play:
        # Watch two games vs a random agent
        eval_action_mask(env_fn, num_games=2, render_mode="human", **env_kwargs)

    print('\007')

    #Calculate training time
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
