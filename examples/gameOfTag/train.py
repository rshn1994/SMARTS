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
import numpy as np
import ray
import os
import yaml

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
from typing import Dict, List


tf1, tf, tfv = try_import_tf()


def info_adapter(obs, reward, info):
    return reward


def action_adapter(model_action):
    throttle, brake, steering = model_action
    # Modify action space limits
    throttle = (throttle + 1) / 2
    brake = (brake + 1) / 2
    # steering = steering
    return np.array([throttle, brake, steering], dtype=np.float32)


def observation_adapter(obs) -> np.ndarray:
    # RGB grid map
    rgb = obs.top_down_rgb.data

    # Replace self color to yellow
    coloured_self = rgb.copy()
    coloured_self[123:132, 126:130, 0] = 255
    coloured_self[123:132, 126:130, 1] = 190
    coloured_self[123:132, 126:130, 2] = 40

    # Convert rgb to grayscale image
    grayscale = rgb2gray(coloured_self)

    # Center frames
    frame = grayscale * 2 - 1
    frame = frame.astype(np.float32)

    # Plot graph
    # fig, axes = plt.subplots(1, 4, figsize=(10, 10))
    # ax = axes.ravel()
    # ax[0].imshow(rgb)
    # ax[0].set_title("RGB")
    # ax[1].imshow(coloured_self)
    # ax[1].set_title("Coloured self - yellow")
    # ax[2].imshow(grayscale, cmap=plt.cm.gray)
    # ax[2].set_title("Grayscale")
    # ax[3].imshow(frame)
    # ax[3].set_title("Centered")
    # fig.tight_layout()
    # plt.show()
    # sys.exit(2)

    return frame


def get_targets(vehicles, target: str):
    target_vehicles = [vehicle for vehicle in vehicles if target in vehicle.id]
    return target_vehicles


def predator_reward_adapter(obs, env_reward):
    reward = 0
    ego = obs.ego_vehicle_state

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 5
        return np.float32(reward)

    # Distance based reward
    targets = get_targets(obs.neighborhood_vehicle_states, "prey")
    if targets:
        # distances = distance_to_targets(ego, targets)
        # min_distance = np.amin(distances)
        # dist_reward = exponential_negative(min_distance)
        # dist_reward = inverse(min_distance)
        # reward += np.clip(dist_reward, 0, 55) / 20  # Reward [0:275]
        reward += 1
    else:  # No neighborhood preys
        #     reward -= 1
        pass

    # Reward for colliding
    for c in obs.events.collisions:
        if "prey" in c.collidee_id:
            reward += 5
            print(f"Predator {ego.id} collided with prey vehicle {c.collidee_id}.")
        else:
            reward -= 5
            print(f"Predator {ego.id} collided with predator vehicle {c.collidee_id}.")

    # Penalty for not moving
    # if obs.events.not_moving:
    # reward -= 2

    return np.float32(reward)


def prey_reward_adapter(obs, env_reward):
    reward = 0
    ego = obs.ego_vehicle_state

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 5
        return np.float32(reward)

    # Distance based reward
    targets = get_targets(obs.neighborhood_vehicle_states, "predator")
    if targets:
        # distances = distance_to_targets(ego, targets)
        # ave_distance = np.average(distances)
        # dist_reward = exponential_positive(ave_distance)
        # dist_reward = linear(ave_distance)
        # reward += np.clip(dist_reward, 0, 55) / 20  # Reward [0:275]
        reward -= 1
    else:  # No neighborhood predators
        # reward += 1
        pass

    # Penalty for colliding
    for c in obs.events.collisions:
        if "predator" in c.collidee_id:
            reward -= 5
            print(f"Prey {ego.id} collided with predator vehicle {c.collidee_id}.")
        else:
            reward -= 5
            print(f"Prey {ego.id} collided with prey vehicle {c.collidee_id}.")

    # Penalty for not moving
    # if obs.events.not_moving:
    #     reward -= 2

    return np.float32(reward)



# def main23(config):

#     print("[INFO] Train")
#     # Save and eval interval
#     save_interval = config["model_para"].get("save_interval", 50)

#     # Mode: Evaluation or Testing
#     mode = Mode(config["model_para"]["mode"])

#     # Traning parameters
#     num_train_epochs = config["model_para"]["num_train_epochs"]
#     batch_size = config["model_para"]["batch_size"]
#     max_batch = config["model_para"]["max_batch"]
#     clip_value = config["model_para"]["clip_value"]
#     critic_loss_weight = config["model_para"]["critic_loss_weight"]
#     ent_discount_val = config["model_para"]["entropy_loss_weight"]
#     ent_discount_rate = config["model_para"]["entropy_loss_discount_rate"]

#     # Create env
#     print("[INFO] Creating environments")
#     seed = config["env_para"]["seed"]
#     # seed = random.randint(0, 4294967295)  # [0, 2^32 -1)
#     env = got_env.TagEnv(config, seed)

#     # Create agent
#     print("[INFO] Creating agents")
#     all_agents = {
#         name: got_agent.TagAgent(name, config)
#         for name in config["env_para"]["agent_ids"]
#     }
#     all_predators_id = env.predators
#     all_preys_id = env.preys

#     # Create model
#     print("[INFO] Creating model")
#     ppo_predator = got_ppo.PPO(AgentType.PREDATOR.value, config)
#     ppo_prey = got_ppo.PPO(AgentType.PREY.value, config)

#     # def interrupt(*args):
#     #     nonlocal mode
#     #     if mode == Mode.TRAIN:
#     #         ppo_predator.save(-1)
#     #         ppo_prey.save(-1)
#     #         policies.save({AgentType.PREDATOR.value:-1, AgentType.PREY.value:-1})
#     #     policies.close()
#     #     env.close()
#     #     print("Interrupt key detected.")
#     #     sys.exit(0)

#     # # Catch keyboard interrupt and terminate signal
#     # signal.signal(signal.SIGINT, interrupt)

#     print("[INFO] Batch loop")
#     states_t = env.reset()
#     episode = 0
#     steps_t = 0
#     episode_reward_predator = 0
#     episode_reward_prey = 0
#     for batch_num in range(max_batch):
#         [agent.reset() for _, agent in all_agents.items()]
#         active_agents = {}

#         print(f"[INFO] New batch data collection {batch_num}/{max_batch}")
#         for cur_step in range(batch_size):

#             # Update all agents which were active in this batch
#             active_agents.update({agent_id: True for agent_id, _ in states_t.items()})

#             # Predict and value action given state
#             actions_t = {}
#             action_samples_t = {}
#             values_t = {}
#             (
#                 actions_t_predator,
#                 action_samples_t_predator,
#                 values_t_predator,
#             ) = ppo_predator.act(states_t)
#             actions_t_prey, action_samples_t_prey, values_t_prey = ppo_prey.act(
#                 states_t
#             )
#             actions_t.update(actions_t_predator)
#             actions_t.update(actions_t_prey)
#             action_samples_t.update(action_samples_t_predator)
#             action_samples_t.update(action_samples_t_prey)
#             values_t.update(values_t_predator)
#             values_t.update(values_t_prey)


#             # Sample action from a distribution
#             action_numpy_t = {
#                 vehicle: action_sample_t.numpy()[0]
#                 for vehicle, action_sample_t in action_samples_t.items()
#             }
#             next_states_t, rewards_t, dones_t, _ = env.step(action_numpy_t)
#             steps_t += 1

#             # Store state, action and reward
#             for agent_id, _ in states_t.items():
#                 all_agents[agent_id].add_trajectory(
#                     action=action_samples_t[agent_id],
#                     value=values_t[agent_id].numpy()[0],
#                     state=states_t[agent_id],
#                     done=int(dones_t[agent_id]),
#                     prob=actions_t[agent_id],
#                     reward=rewards_t[agent_id],
#                 )
#                 if "predator" in agent_id:
#                     episode_reward_predator += rewards_t[agent_id]
#                 else:
#                     episode_reward_prey += rewards_t[agent_id]
#                 if dones_t[agent_id] == 1:
#                     # Remove done agents
#                     del next_states_t[agent_id]
#                     # Print done agents
#                     print(
#                         f"   Done: {agent_id}. Cur_Step: {cur_step}. Step: {steps_t}."
#                     )

#             # Reset when episode completes
#             if dones_t["__all__"]:
#                 # Next episode
#                 next_states_t = env.reset()
#                 episode += 1

#                 # Log rewards
#                 print(
#                     f"   Episode: {episode}. Cur_Step: {cur_step}. "
#                     f"Episode reward predator: {episode_reward_predator}, "
#                     f"Episode reward prey: {episode_reward_prey}."
#                 )
#                 with ppo_predator.tb.as_default():
#                     tf.summary.scalar(
#                         "episode_reward_predator", episode_reward_predator, episode
#                     )
#                 with ppo_prey.tb.as_default():
#                     tf.summary.scalar(
#                         "episode_reward_prey", episode_reward_prey, episode
#                     )

#                 # Reset counters
#                 episode_reward_predator = 0
#                 episode_reward_prey = 0
#                 steps_t = 0

#             # Assign next_states to states
#             states_t = next_states_t

#         # Compute and store last state value
#         for agent_id in active_agents.keys():
#             if dones_t.get(agent_id, None) == 0:  # Agent not done yet
#                 if AgentType.PREDATOR.value in agent_id:
#                     _, _, next_values_t = ppo_predator.act(
#                         {agent_id: next_states_t[agent_id]}
#                     )
#                 elif AgentType.PREY.value in agent_id:
#                     _, _, next_values_t = ppo_prey.act(
#                         {agent_id: next_states_t[agent_id]}
#                     )
#                 else:
#                     raise Exception(f"Unknown {agent_id}.")
#                 all_agents[agent_id].add_last_transition(
#                     value=next_values_t[agent_id].numpy()[0]
#                 )
#             else:  # Agent done
#                 all_agents[agent_id].add_last_transition(value=np.float32(0))

#         # Compute generalised advantages
#         for agent_id in active_agents.keys():
#             all_agents[agent_id].compute_advantages()
#             probs_softmax = tf.nn.softmax(all_agents[agent_id].probs)
#             all_agents[agent_id].probs_softmax = probs_softmax
#             actions = tf.squeeze(all_agents[agent_id].actions, axis=1)
#             action_inds = tf.stack(
#                 [tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis=1
#             )
#             all_agents[agent_id].action_inds = action_inds

#         print("[INFO] Record metrics")
#         # Record predator performance
#         records = []
#         records.append(("predator_tot_loss", np.mean(predator_total_loss), step))
#         records.append(("predator_critic_loss", np.mean(predator_critic_loss), step))
#         records.append(("predator_actor_loss", np.mean(predator_actor_loss), step))
#         records.append(("predator_entropy_loss", np.mean(predator_entropy_loss), step))
#         ppo_predator.write_to_tb(records)

#         # Record prey perfromance
#         records = []
#         records.append(("prey_tot_loss", np.mean(prey_total_loss), step))
#         records.append(("prey_critic_loss", np.mean(prey_critic_loss), step))
#         records.append(("prey_actor_loss", np.mean(prey_actor_loss), step))
#         records.append(("prey_entropy_loss", np.mean(prey_entropy_loss), step))
#         ppo_prey.write_to_tb(records)

#         # Save model
#         if batch_num % save_interval == 0:
#             print("[INFO] Saving model")
#             ppo_predator.save(step)
#             ppo_prey.save(step)
#             policies.save({AgentType.PREDATOR:step, AgentType.PREY:step})

#     # Close policies and env
#     policies.close()
#     env.close()


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


    # This action space should match the input to the action_adapter(..) function below.
    ACTION_SPACE = gym.spaces.Box(
        low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
    )


    # This observation space should match the output of observation_adapter(..) below
    OBSERVATION_SPACE = gym.spaces.Dict(
        {
            "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
            "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        }
    )

    # ------------------------------------------------------------------------------------------
    predators = []
    preys = []
    for agent_id in config["env_para"]["agent_ids"]:
        if "predator" in agent_id:
            predators.append(agent_id)
        elif "prey" in agent_id:
            preys.append(agent_id)
        else:
            raise ValueError(
                f"Expected agent_id to have prefix of 'predator' or 'prey', but got {agent_id}."
            )
    neighborhood_radius = config["env_para"]["neighborhood_radius"]
    rgb_wh = config["env_para"]["rgb_wh"]

    predator_interface = smarts_agent_interface.AgentInterface(
        max_episode_steps=config["env_para"]["max_episode_steps"],
        neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
            radius=neighborhood_radius
        ),
        rgb=smarts_agent_interface.RGB(
            width=256, height=256, resolution=rgb_wh / 256
        ),
        vehicle_color="Blue",
        action=getattr(smarts_controllers.ActionSpaceType, "Continuous"),
        done_criteria=smarts_agent_interface.DoneCriteria(
            collision=False,
            off_road=True,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
            not_moving=False,
            agents_alive=smarts_agent_interface.AgentsAliveDoneCriteria(
                agent_lists_alive=[
                    smarts_agent_interface.AgentsListAlive(
                        agents_list=preys, minimum_agents_alive_in_list=1
                    ),
                ]
            ),
        ),
    )

    prey_interface = smarts_agent_interface.AgentInterface(
        max_episode_steps=config["env_para"]["max_episode_steps"],
        neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
            radius=neighborhood_radius
        ),
        rgb=smarts_agent_interface.RGB(
            width=256, height=256, resolution=rgb_wh / 256
        ),
        vehicle_color="White",
        action=getattr(smarts_controllers.ActionSpaceType, "Continuous"),
        done_criteria=smarts_agent_interface.DoneCriteria(
            collision=True,
            off_road=True,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
            not_moving=False,
            agents_alive=smarts_agent_interface.AgentsAliveDoneCriteria(
                agent_lists_alive=[
                    smarts_agent_interface.AgentsListAlive(
                        agents_list=predators, minimum_agents_alive_in_list=1
                    ),
                ]
            ),
        ),
    )

    # Create agent spec
    agent_specs = {
        agent_id: smarts_agent.AgentSpec(
            interface=predator_interface,
            agent_builder=got_agent.TagAgent,
            observation_adapter=observation_adapter,
            reward_adapter=predator_reward_adapter,
            action_adapter=action_adapter,
            info_adapter=info_adapter,
        )
        if "predator" in agent_id
        else smarts_agent.AgentSpec(
            interface=prey_interface,
            agent_builder=got_agent.TagAgent,
            observation_adapter=observation_adapter,
            reward_adapter=prey_reward_adapter,
            action_adapter=action_adapter,
            info_adapter=info_adapter,
        )
        for agent_id in config["env_para"]["agent_ids"]
    }




    ray.init(num_cpus = mp.cpu_count()-2 if mp.cpu_count()-2 > 0 else None)

    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config=dict(
            extra_config,
            **{
                "env": "BreakoutNoFrameskip-v4"
                if args.use_vision_network
                else "CartPole-v0",
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                "callbacks": {
                    "on_train_result": check_has_custom_metric,
                },
                "model": {
                    "custom_model": "keras_q_model"
                    if args.run == "DQN"
                    else "keras_model"
                },
                "framework": "tf",
            }
        ),
    )

    trainer = pg.PGAgent(
        env=RLlibHiWayEnv,
        config={
            "multiagent": {
                "policies": {
                    "predator": (
                        None,
                        predator_obs_space,
                        predator_act_space,
                        {"gamma": 0.85},
                    ),
                    "prey": (None, prey_obs_space, prey_act_space, {"gamma": 0.99}),
                },
                "policy_mapping_fn": lambda agent_id: "predator"
                if agent_id.startswith("predator")
                else "prey",
            },
        },
    )

    tune.run(
        run_or_experiment=PPOTrainer, 
        config = {
            "env": RLlibHiWayEnv,
            "log_level": "WARN",
            "num_workers": num_workers,
            "env_config": {
                "scenarios": [str(Path(scenario).expanduser().resolve().absolute())],
                "headless": headless,
                "agent_specs": agent_specs,
            },
            "multiagent": {"policies": rllib_policies},
            "callbacks": Callbacks,
        }
    )


