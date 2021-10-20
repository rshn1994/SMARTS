import sys
import signal
import stable_baselines3 as sb3
import yaml

from datetime import datetime
from examples.gameOfTag import env as got_env
from examples.gameOfTag.types import Mode
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env(config, rank):
    def _init():
        env = got_env.SingleAgent(config, rank)
        return env

    set_random_seed(config["env_para"]["seed"], using_cuda=True)
    return _init


def main(config):

    mode = config["model_para"]["mode"]

    print("[INFO] Check Env")
    env_single = make_env(config, 0)()
    try:
        env_checker.check_env(env_single)
        print("Completed environment check")
    except Exception as e:
        print(f"Your environment is not single-agent gym compliant.")
        raise e
    finally:
        env_single.close()

    # Create the vectorized environment
    if mode == Mode.TRAIN:
        num_env = config["env_para"]["num_env"]
        env = SubprocVecEnv([make_env(config, i) for i in range(num_env)])
    else:
        env = make_env(config, 0)()

    # Tensorboard
    tb_path = Path(config["model_para"]["tensorboard_path"]).joinpath(
        f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    )

    # Model
    print("[INFO] PPO Model")
    if config["model_para"]["model_initial"]:  # Start from existing model
        print("[INFO] Load Model")
        model = PPO.load(config["model_para"]["model_agent"])
    else:  # Start from new model
        print("[INFO] New Model")
        model = PPO("CnnPolicy", env, ent_coef=config["model_para"]["entropy_loss_weight"], tensorboard_log=tb_path, verbose=1)
        # model = PPO("CnnPolicy", env, tensorboard_log=tb_path, verbose=1)

    print("[INFO] Interrupt Handler")

    def interrupt(*args):
        nonlocal model
        model.save(get_model_path(config))
        env.close()
        print("Interrupt key detected.")
        sys.exit(0)

    if mode == Mode.TRAIN:
        # Catch keyboard interrupt and terminate signal
        signal.signal(signal.SIGINT, interrupt)

        # Train
        print("[INFO] Train")
        model.learn(
            total_timesteps=config["model_para"]["max_time_steps"], log_interval=1
        )
        model.save(get_model_path(config))

        print("[INFO] Wait")
        # import time
        # time.sleep(834)

        print("[INFO] Delete Model")
        del model  # remove to demonstrate saving and loading

    if mode == Mode.EVALUATE:
        print("[INFO] Evaluate")
        obs = env.reset()
        dones = False
        while True:
            if dones:
                obs = env.reset()
                print("Env reset")
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

    print("[INFO] Close Env")
    # Close env
    env.close()


def get_model_path(config):
    return Path(config["model_para"]["model_path"]).joinpath(
        f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    )


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
