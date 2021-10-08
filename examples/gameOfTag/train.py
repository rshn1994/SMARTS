import sys
import signal
import stable_baselines3 as sb3
import yaml

from datetime import datetime
from examples.gameOfTag import env as got_env
from examples.gameOfTag.types import Mode
from pathlib import Path
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


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
    # # Tensorboard
    # path = Path(config["model_para"]["tensorboard_path"]).joinpath(
    #     f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    # )
    # tb = tf.summary.create_file_writer(str(path))

    # SB3 environments
    # env = make_vec_env("CartPole-v1", n_envs=4)

    try:
        env_checker.check_env(env)
        print("Completed environment check")
    except Exception as e:
        print(f"Your environment is not single-agent gym compliant.")
        raise e

    model = PPO("CnnPolicy", env, verbose=1)
    print("completed model instantiation ??????????????????????")

    def interrupt(*args):
        nonlocal mode
        if mode == Mode.TRAIN:
            model.save(model_path)
        env.close()
        print("Interrupt key detected.")
        sys.exit(0)

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, interrupt)

    # Train
    model.learn(total_timesteps=5)
    model.save(model_path)

    print("completed training ??????????????????????")
    import time
    time.sleep(834)

    del model  # remove to demonstrate saving and loading

    model = PPO.load(model_path)
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

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
