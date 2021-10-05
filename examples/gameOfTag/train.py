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
import multiprocessing as mp
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
from ray.rllib.agents.dqn.distributional_q_tf_model import \
    DistributionalQTFModel
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet import VisionNetwork as MyVisionNetwork
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, \
    LEARNER_STATS_KEY
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from typing import Dict, List


tf1, tf, tfv = try_import_tf()


def main23(config):

    print("[INFO] Train")
    # Save and eval interval
    save_interval = config["model_para"].get("save_interval", 50)

    # Mode: Evaluation or Testing
    mode = Mode(config["model_para"]["mode"])

    # Traning parameters
    num_train_epochs = config["model_para"]["num_train_epochs"]
    batch_size = config["model_para"]["batch_size"]
    max_batch = config["model_para"]["max_batch"]
    clip_value = config["model_para"]["clip_value"]
    critic_loss_weight = config["model_para"]["critic_loss_weight"]
    ent_discount_val = config["model_para"]["entropy_loss_weight"]
    ent_discount_rate = config["model_para"]["entropy_loss_discount_rate"]

    # Create env
    print("[INFO] Creating environments")
    seed = config["env_para"]["seed"]
    # seed = random.randint(0, 4294967295)  # [0, 2^32 -1)
    env = got_env.TagEnv(config, seed)

    # Create agent
    print("[INFO] Creating agents")
    all_agents = {
        name: got_agent.TagAgent(name, config)
        for name in config["env_para"]["agent_ids"]
    }
    all_predators_id = env.predators
    all_preys_id = env.preys

    # Create model
    print("[INFO] Creating model")
    ppo_predator = got_ppo.PPO(AgentType.PREDATOR.value, config)
    ppo_prey = got_ppo.PPO(AgentType.PREY.value, config)

    # Create parallel policies
    policy_constructors = {
        AgentType.PREDATOR.value: lambda: got_ppo.PPO(
            name=AgentType.PREDATOR.value,
            config=config,
        ),
        AgentType.PREY.value: lambda: got_ppo.PPO(
            name=AgentType.PREY.value,
            config=config,
        ),
    }
    policies = ParallelPolicy(
        policy_constructors=policy_constructors,
    )

    # def interrupt(*args):
    #     nonlocal mode
    #     if mode == Mode.TRAIN:
    #         ppo_predator.save(-1)
    #         ppo_prey.save(-1)
    #         policies.save({AgentType.PREDATOR.value:-1, AgentType.PREY.value:-1})
    #     policies.close()
    #     env.close()
    #     print("Interrupt key detected.")
    #     sys.exit(0)

    # # Catch keyboard interrupt and terminate signal
    # signal.signal(signal.SIGINT, interrupt)

    print("[INFO] Batch loop")
    states_t = env.reset()
    episode = 0
    steps_t = 0
    episode_reward_predator = 0
    episode_reward_prey = 0
    for batch_num in range(max_batch):
        [agent.reset() for _, agent in all_agents.items()]
        active_agents = {}

        print(f"[INFO] New batch data collection {batch_num}/{max_batch}")
        for cur_step in range(batch_size):

            # Update all agents which were active in this batch
            active_agents.update({agent_id: True for agent_id, _ in states_t.items()})

            # Predict and value action given state
            actions_t = {}
            action_samples_t = {}
            values_t = {}
            (
                actions_t_predator,
                action_samples_t_predator,
                values_t_predator,
            ) = ppo_predator.act(states_t)
            actions_t_prey, action_samples_t_prey, values_t_prey = ppo_prey.act(
                states_t
            )
            actions_t.update(actions_t_predator)
            actions_t.update(actions_t_prey)
            action_samples_t.update(action_samples_t_predator)
            action_samples_t.update(action_samples_t_prey)
            values_t.update(values_t_predator)
            values_t.update(values_t_prey)

            _, _, values_t_prey_2 = policies.act(states_t)
            print("-----------------------------------")
            print("Sequentially processed")
            print(values_t)
            print("-----------------------------------")
            print("Multiprocessed")
            print(values_t_prey_2)
            print("\n\n")      

            import sys
            sys.exit(1)

            # Sample action from a distribution
            action_numpy_t = {
                vehicle: action_sample_t.numpy()[0]
                for vehicle, action_sample_t in action_samples_t.items()
            }
            next_states_t, rewards_t, dones_t, _ = env.step(action_numpy_t)
            steps_t += 1

            # Store state, action and reward
            for agent_id, _ in states_t.items():
                all_agents[agent_id].add_trajectory(
                    action=action_samples_t[agent_id],
                    value=values_t[agent_id].numpy()[0],
                    state=states_t[agent_id],
                    done=int(dones_t[agent_id]),
                    prob=actions_t[agent_id],
                    reward=rewards_t[agent_id],
                )
                if "predator" in agent_id:
                    episode_reward_predator += rewards_t[agent_id]
                else:
                    episode_reward_prey += rewards_t[agent_id]
                if dones_t[agent_id] == 1:
                    # Remove done agents
                    del next_states_t[agent_id]
                    # Print done agents
                    print(
                        f"   Done: {agent_id}. Cur_Step: {cur_step}. Step: {steps_t}."
                    )

            # Reset when episode completes
            if dones_t["__all__"]:
                # Next episode
                next_states_t = env.reset()
                episode += 1

                # Log rewards
                print(
                    f"   Episode: {episode}. Cur_Step: {cur_step}. "
                    f"Episode reward predator: {episode_reward_predator}, "
                    f"Episode reward prey: {episode_reward_prey}."
                )
                with ppo_predator.tb.as_default():
                    tf.summary.scalar(
                        "episode_reward_predator", episode_reward_predator, episode
                    )
                with ppo_prey.tb.as_default():
                    tf.summary.scalar(
                        "episode_reward_prey", episode_reward_prey, episode
                    )

                # Reset counters
                episode_reward_predator = 0
                episode_reward_prey = 0
                steps_t = 0

            # Assign next_states to states
            states_t = next_states_t

        # Compute and store last state value
        for agent_id in active_agents.keys():
            if dones_t.get(agent_id, None) == 0:  # Agent not done yet
                if AgentType.PREDATOR.value in agent_id:
                    _, _, next_values_t = ppo_predator.act(
                        {agent_id: next_states_t[agent_id]}
                    )
                elif AgentType.PREY.value in agent_id:
                    _, _, next_values_t = ppo_prey.act(
                        {agent_id: next_states_t[agent_id]}
                    )
                else:
                    raise Exception(f"Unknown {agent_id}.")
                all_agents[agent_id].add_last_transition(
                    value=next_values_t[agent_id].numpy()[0]
                )
            else:  # Agent done
                all_agents[agent_id].add_last_transition(value=np.float32(0))

        # Compute generalised advantages
        for agent_id in active_agents.keys():
            all_agents[agent_id].compute_advantages()
            probs_softmax = tf.nn.softmax(all_agents[agent_id].probs)
            all_agents[agent_id].probs_softmax = probs_softmax
            actions = tf.squeeze(all_agents[agent_id].actions, axis=1)
            action_inds = tf.stack(
                [tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis=1
            )
            all_agents[agent_id].action_inds = action_inds

        predator_total_loss = np.zeros((num_train_epochs))
        predator_actor_loss = np.zeros((num_train_epochs))
        predator_critic_loss = np.zeros(((num_train_epochs)))
        predator_entropy_loss = np.zeros((num_train_epochs))

        prey_total_loss = np.zeros((num_train_epochs))
        prey_actor_loss = np.zeros((num_train_epochs))
        prey_critic_loss = np.zeros(((num_train_epochs)))
        prey_entropy_loss = np.zeros((num_train_epochs))

        # Elapsed steps
        step = (batch_num + 1) * batch_size

        if mode == Mode.EVALUATE:
            continue

        print("[INFO] Training")
        # Train predator and prey.
        # Run multiple gradient ascent on the samples.
        for epoch in range(num_train_epochs):
            for agent_id in active_agents.keys():
                agent = all_agents[agent_id]
                if agent_id in all_predators_id:
                    loss_tuple = got_ppo.train_model(
                        model=ppo_predator.model,
                        optimizer=ppo_predator.optimizer,
                        action_inds=agent.action_inds,
                        old_probs=tf.gather_nd(agent.probs_softmax, agent.action_inds),
                        states=agent.states,
                        advantages=agent.advantages,
                        discounted_rewards=agent.discounted_rewards,
                        ent_discount_val=ent_discount_val,
                        clip_value=clip_value,
                        critic_loss_weight=critic_loss_weight,
                    )
                    predator_total_loss[epoch] += loss_tuple[0]
                    predator_actor_loss[epoch] += loss_tuple[1]
                    predator_critic_loss[epoch] += loss_tuple[2]
                    predator_entropy_loss[epoch] += loss_tuple[3]

                if agent_id in all_preys_id:
                    loss_tuple = got_ppo.train_model(
                        model=ppo_prey.model,
                        optimizer=ppo_prey.optimizer,
                        action_inds=agent.action_inds,
                        old_probs=tf.gather_nd(agent.probs_softmax, agent.action_inds),
                        states=agent.states,
                        advantages=agent.advantages,
                        discounted_rewards=agent.discounted_rewards,
                        ent_discount_val=ent_discount_val,
                        clip_value=clip_value,
                        critic_loss_weight=critic_loss_weight,
                    )
                    prey_total_loss[epoch] += loss_tuple[0]
                    prey_actor_loss[epoch] += loss_tuple[1]
                    prey_critic_loss[epoch] += loss_tuple[2]
                    prey_entropy_loss[epoch] += loss_tuple[3]

        ent_discount_val *= ent_discount_rate

        print("[INFO] Record metrics")
        # Record predator performance
        records = []
        records.append(("predator_tot_loss", np.mean(predator_total_loss), step))
        records.append(("predator_critic_loss", np.mean(predator_critic_loss), step))
        records.append(("predator_actor_loss", np.mean(predator_actor_loss), step))
        records.append(("predator_entropy_loss", np.mean(predator_entropy_loss), step))
        ppo_predator.write_to_tb(records)

        # Record prey perfromance
        records = []
        records.append(("prey_tot_loss", np.mean(prey_total_loss), step))
        records.append(("prey_critic_loss", np.mean(prey_critic_loss), step))
        records.append(("prey_actor_loss", np.mean(prey_actor_loss), step))
        records.append(("prey_entropy_loss", np.mean(prey_entropy_loss), step))
        ppo_prey.write_to_tb(records)

        # # Evaluate model
        # if batch_num % eval_interval == 0:
        #     print("[INFO] Running evaluation...")
        #     (
        #         avg_reward_predator,
        #         avg_reward_prey,
        #     ) = evaluate.evaluate(ppo_predator, ppo_prey, config)

        # Save model
        if batch_num % save_interval == 0:
            print("[INFO] Saving model")
            ppo_predator.save(step)
            ppo_prey.save(step)
            policies.save({AgentType.PREDATOR:step, AgentType.PREY:step})

    # Close policies and env
    policies.close()
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


    ray.init(num_cpus=mp.cpu_count()-2 or None)

    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config=dict(
            extra_config,
            **{
                "env": "BreakoutNoFrameskip-v4"
                if args.use_vision_network else "CartPole-v0",
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                "callbacks": {
                    "on_train_result": check_has_custom_metric,
                },
                "model": {
                    "custom_model": "keras_q_model"
                    if args.run == "DQN" else "keras_model"
                },
                "framework": "tf",
            }))

    trainer = pg.PGAgent(
        env=RLlibHiWayEnv, 
        config={
        "multiagent": {
            "policies": {
                # the first tuple value is None -> uses default policy
                "car1": (None, car_obs_space, car_act_space, {"gamma": 0.85}),
                "car2": (None, car_obs_space, car_act_space, {"gamma": 0.99}),
                "traffic_light": (None, tl_obs_space, tl_act_space, {}),
            },
            "policy_mapping_fn":
                lambda agent_id:
                    "traffic_light"  # Traffic lights are always controlled by this policy
                    if agent_id.startswith("traffic_light_")
                    else random.choice(["car1", "car2"])  # Randomly choose from car policies
        },
    })


    tune.run(PPOTrainer, config={"env": "CartPole-v0", "train_batch_size": 4000})
