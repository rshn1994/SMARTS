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


class PPO(RL):
    def __init__(self, name, config):
        super(PPO, self).__init__()

        # Tensorboard
        path = Path(self.config["model_para"]["tensorboard_path"]).joinpath(
            f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        )
        self.tb = tf.summary.create_file_writer(str(path))

    def save(self, version: int):
        save_path = self.model_path / str(version)
        tf.keras.models.save_model(
            model=self.model,
            filepath=save_path,
        )

    def write_to_tb(self, records):
        with self.tb.as_default():
            for name, value, step in records:
                tf.summary.scalar(name, value, step)
