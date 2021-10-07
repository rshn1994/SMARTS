# import os

# # Set pythonhashseed
# os.environ["PYTHONHASHSEED"] = "0"
# # Silence the logs of TF
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # The below is necessary for starting Numpy generated random numbers
# # in a well-defined initial state.
# import numpy as np

# np.random.seed(123)

# # The below is necessary for starting core Python generated random numbers
# # in a well-defined state.
# import random as python_random

# python_random.seed(123)

# # The below set_seed() will make random number generation
# # in the TensorFlow backend have a well-defined initial state.
# # For further details, see:
# # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
# import tensorflow as tf

# tf.random.set_seed(123)
# --------------------------------------------------------------------------

import argparse
import gym
import multiprocessing as mp
import ray
import os
import sys
import signal
import yaml

from datetime import datetime
from enum import Enum
from examples.gameOfTag import env as got_env
from examples.gameOfTag import agent as got_agent
from examples.gameOfTag import ppo as got_ppo
from examples.gameOfTag.types import AgentType, Mode
from pathlib import Path
from ray import tune
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet import VisionNetwork as MyVisionNetwork
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
from smarts.env.wrappers import frame_stack as smarts_frame_stack
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import Dict, List


tf1, tf, tfv = try_import_tf()

def _load(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )

def main(config):

    name = config["env_para"]["env_name"]
    save_interval = config["model_para"].get("save_interval", 50)
    mode = Mode(config["model_para"]["mode"])
    num_train_epochs = config["model_para"]["num_train_epochs"]
    batch_size = config["model_para"]["batch_size"]
    max_batch = config["model_para"]["max_batch"]
    clip_value = config["model_para"]["clip_value"]
    critic_loss_weight = config["model_para"]["critic_loss_weight"]
    ent_discount_val = config["model_para"]["entropy_loss_weight"]
    ent_discount_rate = config["model_para"]["entropy_loss_discount_rate"]
    seed = config["env_para"]["seed"]
    
    env = got_env.SingleEnv(config)

    # Model
    model_path = Path(config["model_para"]["model_path"]).joinpath(
        f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    )
    # Tensorboard
    path = Path(config["model_para"]["tensorboard_path"]).joinpath(
        f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    )
    tb = tf.summary.create_file_writer(str(path))


    # SB3 environments
    env = make_vec_env("CartPole-v1", n_envs=4)
    model = PPO("CnnPolicy", env, verbose=1)

    def interrupt(*args):
        nonlocal mode
        if mode == Mode.TRAIN:
            model.save("ppo_single_agent")
        env.close()
        print("Interrupt key detected.")
        sys.exit(0)

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, interrupt)

    # Train
    model.learn(total_timesteps=5)
    model.save(model_path)

    del model # remove to demonstrate saving and loading

    model = PPO.load(model_path)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    # Close env
    env.close()



if __name__ == "__main__":
    config_yaml = (Path(__file__).absolute().parent).joinpath("got.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Setup GPU
    # gpus = tf.config.list_physical_devices("GPU")
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices("GPU")
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
    # else:
    #     warnings.warn(
    #         f"Not configured to use GPU or GPU not available.",
    #         ResourceWarning,
    #     )
    #     # raise SystemError("GPU device not found")

    main(config)

