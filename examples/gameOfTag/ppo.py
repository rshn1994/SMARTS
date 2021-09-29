import os

# Set pythonhashseed
os.environ["PYTHONHASHSEED"] = "0"
# Silence the logs of TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
import numpy as np

np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
import random as python_random

python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
import tensorflow as tf

tf.random.set_seed(123)

# --------------------------------------------------------------------------

import absl.logging
import tensorflow_probability as tfp

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Suppress warning
absl.logging.set_verbosity(absl.logging.ERROR)


class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        """
        Args:
            num_actions (int): Number of continuous actions to output
        """
        super(NeuralNetwork, self).__init__()
        self.num_actions = num_actions
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=32,
            strides=(4, 4),
            padding="valid",
            activation=tf.keras.activations.relu,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=17,
            strides=(2, 2),
            padding="valid",
            activation=tf.keras.activations.relu,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=128, activation=tf.keras.activations.relu
        )
        self.dense_value = tf.keras.layers.Dense(
            units=64, activation=tf.keras.activations.relu
        )
        self.dense_policy = tf.keras.layers.Dense(
            units=64, activation=tf.keras.activations.relu
        )
        self.policy = tf.keras.layers.Dense(units=self.num_actions)
        self.value = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        """
        Args:
            inputs ([batch_size, width, height, depth]): Input images to predict actions for.

        Returns:
            [type]: 2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :] represents the unnormalized "log-probabilities" for all classes.
            [type]: Value of state
        """
        conv1_out = self.conv1(inputs)
        conv2_out = self.conv2(conv1_out)
        flatten_out = self.flatten(conv2_out)
        dense1_out = self.dense1(flatten_out)
        dense_policy_out = self.dense_policy(dense1_out)
        dense_value_out = self.dense_value(dense1_out)
        policy = self.policy(dense_policy_out)
        value = self.value(dense_value_out)
        return policy, value


class RL:
    def __init__(self):
        pass

    def act(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class PPO(RL):
    def __init__(self, name, config):
        super(PPO, self).__init__()

        self.name = name
        self.config = config
        self.seed = config["env_para"]["seed"]
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["model_para"]["initial_lr_" + self.name]
        )

        # Model
        self.model = None
        if self.config["model_para"]["model_initial"]:  # Start from existing model
            self.model = _load(self.config["model_para"]["model_" + self.name])
        else:  # Start from new model
            self.model = NeuralNetwork(self.config["model_para"]["action_dim"])
        # Path for newly trained model
        self.model_path = Path(self.config["model_para"]["model_path"]).joinpath(
            f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        )

        # Tensorboard
        path = Path(self.config["model_para"]["tensorboard_path"]).joinpath(
            f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        )
        self.tb = tf.summary.create_file_writer(str(path))

    def close(self):
        pass

    def save(self, version: int):
        save_path = self.model_path / str(version)
        tf.keras.models.save_model(
            model=self.model,
            filepath=save_path,
        )

    def act(self, obs):
        actions = {}
        action_samples = {}
        values = {}

        ordered_obs = _dict_to_ordered_list(obs)

        # NOTE: Order of items drawn from dict may affect reproducibility of this
        # function due to order of sampling by `actions_dist_t.sample()`.
        for vehicle, state in ordered_obs:
            if self.name in vehicle:
                print(vehicle,"name matched -----------------")
                print(type(state), state.shape, "---------------")
                actions_t, values_t = self.model(
                    np.expand_dims(state, axis=0)
                )
                print("After actions_t !!!!!!!!!!!!!!!")
                actions_dist_t = tfp.distributions.Categorical(logits=actions_t)

                actions[vehicle] = tf.squeeze(actions_t, axis=0)
                action_samples[vehicle] = tf.squeeze(
                    actions_dist_t.sample(1, seed=self.seed), axis=0
                )
                values[vehicle] = tf.squeeze(values_t, axis=0)
        return actions, action_samples, values

    def write_to_tb(self, records):
        with self.tb.as_default():
            for name, value, step in records:
                tf.summary.scalar(name, value, step)


def _dict_to_ordered_list(dic: Dict[str, Any]) -> List[Tuple[str, Any]]:

    li = [tuple(x) for x in dic.items()]
    li.sort(key=lambda tup: tup[0])  # Sort in-place

    return li


def _load(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"NeuralNetwork": NeuralNetwork},
        compile=False,
    )


def train_model(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers,
    action_inds: tf.TensorSpec(shape=(None, 2), dtype=tf.dtypes.int32),
    old_probs,
    states: List[np.ndarray],
    advantages: np.ndarray,
    discounted_rewards: np.ndarray,
    ent_discount_val: float,
    clip_value: float,
    critic_loss_weight: float,
):
    # -> Tuple[
    #     tf.TensorSpec(shape=(), dtype=tf.dtypes.float32),
    #     tf.TensorSpec(shape=(), dtype=tf.dtypes.float32),
    #     tf.TensorSpec(shape=(), dtype=tf.dtypes.float32),
    #     tf.TensorSpec(shape=(), dtype=tf.dtypes.float32),
    # ]:

    with tf.GradientTape() as tape:
        policy_logits, values = model.call(tf.stack(states))
        act_loss = actor_loss(
            advantages, old_probs, action_inds, policy_logits, clip_value
        )
        cri_loss = critic_loss(discounted_rewards, values, critic_loss_weight)
        ent_loss = entropy_loss(policy_logits, ent_discount_val)
        tot_loss = act_loss + cri_loss + ent_loss

    grads = tape.gradient(tot_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return tot_loss, act_loss, cri_loss, ent_loss


# Clipped objective term, to be maximized
# @tf.function(experimental_relax_shapes=True)
def actor_loss(advantages, old_probs, action_inds, policy_logits, clip_value):
    probs = tf.nn.softmax(policy_logits)
    new_probs = tf.gather_nd(probs, action_inds)
    ratio = new_probs / old_probs  # Ratio is always positive

    policy_loss = -tf.reduce_mean(  # -Expectation
        tf.math.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1.0 - clip_value, 1.0 + clip_value) * advantages,
        )
    )
    return policy_loss


# Entropy term to encourage exploration, to be maximized
# @tf.function(experimental_relax_shapes=True)
def entropy_loss(policy_logits, ent_discount_val):
    probs = tf.nn.softmax(policy_logits)
    entropy_loss = -tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(probs, probs)
    )
    return entropy_loss * ent_discount_val


# Error term on value estimation, to be minimized
# @tf.function(experimental_relax_shapes=True)
def critic_loss(discounted_rewards, value_est, critic_loss_weight):
    return (
        tf.reduce_mean(
            tf.keras.losses.mean_squared_error(discounted_rewards, value_est)
        )
        * critic_loss_weight
    )
