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

import multiprocessing as mp
import numpy as np
import sys
import warnings

from examples.gameOfTag.types import AgentType
from collections import defaultdict
from ppo import RL
from smarts.env.utils.cloud_pickle import CloudpickleWrapper
from typing import Any, Callable, Dict, Tuple


__all__ = ["ParallelPolicy"]


PolicyConstructor = Callable[[], RL]


class ParallelPolicy:
    """Batch together multiple policies and step them in parallel. Each
    policy is simulated in an external process for lock-free parallelism
    using `multiprocessing` processes, and pipes for communication.

    Note:
        Simulation might slow down when number of parallel environments
        requested exceed number of available CPU logical cores.
    """

    def __init__(
        self,
        policy_constructors: Dict[str, PolicyConstructor],
    ):
        """The policies can be different but must use the same input and output interfaces.

        Args:
            policy_constructors (Dict[str, PolicyConstructor]): List of callables that create policies.
        """

        if len(policy_constructors) > mp.cpu_count():
            warnings.warn(
                f"Simulation might slow down, since the requested number of parallel "
                f"policies ({len(policy_constructors)}) exceed the number of available "
                f"CPU cores ({mp.cpu_count()}).",
                ResourceWarning,
            )

        if any([not callable(ctor) for _, ctor in policy_constructors.items()]):
            raise TypeError(
                f"Found non-callable `policy_constructors`. Expected `policy_constructors` of type "
                f"`Dict[str, Callable[[], RL]]`, but got {policy_constructors})."
            )

        # Worker polling period in seconds.
        self._polling_period = 0.1

        self.closed = False
        self.policy_ids = policy_constructors.keys()
 
        # Fork is not a thread safe method
        forkserver_available = 'forkserver' in mp.get_all_start_methods()
        start_method = 'forkserver' if forkserver_available else 'spawn'
        mp_ctx = mp.get_context(start_method)
 
        self.error_queue = mp_ctx.Queue()
        self.parent_pipes = {}
        self.processes = {}
        for policy_id, policy_constructor in policy_constructors.items():
            parent_pipe, child_pipe = mp_ctx.Pipe()
            process = mp_ctx.Process(
                target=_worker,
                name=f"Worker-<{type(self).__name__}>-<{policy_id}>",
                args=(
                    CloudpickleWrapper(policy_constructor),
                    child_pipe,
                    self._polling_period,
                ),
            )
            self.parent_pipes.update({policy_id: parent_pipe})
            self.processes.update({policy_id: process})

            # Daemonic subprocesses quit when parent process quits. However, daemonic
            # processes cannot spawn children. Hence, `process.daemon` is set to False.
            process.daemon = False
            process.start()
            child_pipe.close()

        # Wait for all policies to successfully startup
        results = {
            policy_id: pipe.recv() for policy_id, pipe in self.parent_pipes.items()
        }
        self._raise_if_errors(results)

    @staticmethod
    def _agent_to_policy_id(agent_id:str, policy_ids)->str:
        ids = [policy_id for policy_id in policy_ids if policy_id in agent_id]
        if len(ids) == 0:
            raise KeyError(f"Agent_id {agent_id} does not match any policy ids {policy_ids}.")
        if len(ids) > 1:
            raise KeyError(f"Agent_id {agent_id} matches multiple policy ids {policy_ids}.")
        return ids[0]

    def act(self, states_t: Dict[str, Any]) -> Dict[str, Any]:
        dd_state = defaultdict(dict)
        for agent_id, state in states_t.items():
            policy_id = self._agent_to_policy_id(agent_id, self.policy_ids)
            dd_state[policy_id].update({agent_id: state})

        for policy_id, states in dd_state.items():
            self.parent_pipes[policy_id].send(("act", states))
  
        results = {
            policy_id: self.parent_pipes[policy_id].recv()
            for policy_id in dd_state.keys()
        }
        self._raise_if_errors(results)

        actions_t = {}
        action_samples_t = {}
        values_t = {}
        for policy_id, (result, _) in results.items():
            actions, action_samples, values = result
            actions_t.update(actions)
            action_samples_t.update(action_samples)
            values_t.update(values)

        return actions_t, action_samples_t, values_t


    def train(self, states: Dict[str, Any]) -> Dict[str, Any]:
        # for agent_id, states_t in states.items():
        #     policy_id = self._agent_to_policy_id(agent_id)
        #     self.parent_pipes[policy_id].send(("act", states_t))

        # results = {
        #     policy_id: self.parent_pipes[policy_id].recv()
        #     for policy_id in states.keys()
        # }
        # self._raise_if_errors(results)

        return


    def save(self, versions: Dict[str, int]):
        """Save the current policy.

        Args:
            versions (Dict[str, int]): A dictionary, with the key being the policy id and the value
                being the version number for the current policy model to be saved.
        """
        for policy_id, version in versions.items():
            self.parent_pipes[policy_id].send(("save", version))

        results = {
            policy_id: self.parent_pipes[policy_id].recv()
            for policy_id in versions.keys()
        }
        self._raise_if_errors(results)

    def write_to_tb(self, records: Dict[str, Any]):
        """Write records to tensorboard.

        Args:
            records (Dict[str, Any]): A dictionary, with the key being the policy id and the value
                being the data to be recorded for that policy.
        """
        for policy_id, record in records.items():
            self.parent_pipes[policy_id].send(("write_to_tb", record))

        results = {
            policy_id: self.parent_pipes[policy_id].recv()
            for policy_id in records.keys()
        }
        self._raise_if_errors(results)

    def _raise_if_errors(self, results: Dict[str, Tuple[Any, bool]]):
        successes = list(zip(*results.values()))[1]
        if all(successes):
            return

        for policy_id, (error, success) in results.items():
            if not success:
                exctype, value = error
                print(
                    f"Exception in Worker-<{type(self).__name__}>-<{policy_id}>: {exctype.__name__}\n  {value}"
                )

        self.close()
        raise Exception("Error in parallel policy workers.")

    def close(self, terminate=False):
        """Closes all processes alive.

        Args:
            terminate (bool, optional): If `True`, then the `close` operation is forced and all
                processes are terminated. Defaults to False.
        """
        if terminate:
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes.values():
                try:
                    pipe.send(("close", None))
                    pipe.close()
                except IOError:
                    # The connection was already closed.
                    pass

        for process in self.processes.values():
            process.join()

        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()


# def _train(
#     num_train_epochs: int,
#     agents: List[got_agent.TagAgent],
#     rl_algo: got_ppo.PPO,
#     ent_discount_val: float,
#     clip_value: float,
#     critic_loss_weight: float,
# ):

#     total_loss = np.zeros((num_train_epochs))
#     actor_loss = np.zeros((num_train_epochs))
#     critic_loss = np.zeros(((num_train_epochs)))
#     entropy_loss = np.zeros((num_train_epochs))

#     for epoch in range(num_train_epochs):
#         for agent in agents:
#             loss_tuple = got_ppo.train_model(
#                 model=rl_algo.model,
#                 optimizer=rl_algo.optimizer,
#                 action_inds=agent.action_inds,
#                 old_probs=tf.gather_nd(agent.probs_softmax, agent.action_inds),
#                 states=agent.states,
#                 advantages=agent.advantages,
#                 discounted_rewards=agent.discounted_rewards,
#                 ent_discount_val=ent_discount_val,
#                 clip_value=clip_value,
#                 critic_loss_weight=critic_loss_weight,
#             )
#             total_loss[epoch] += loss_tuple[0]
#             actor_loss[epoch] += loss_tuple[1]
#             critic_loss[epoch] += loss_tuple[2]
#             entropy_loss[epoch] += loss_tuple[3]

#     return total_loss, actor_loss, critic_loss, entropy_loss


def _worker(
    policy_constructor: CloudpickleWrapper,
    pipe: mp.connection.Connection,
    polling_period: float = 0.1,
):
    """Process to build and run a policy. Using a pipe to communicate with parent, the
    process receives instructions, and returns results.

    Args:
        policy_constructor (CloudpickleWrapper): Callable which constructs the policy.
        pipe (mp.connection.Connection): Child's end of the pipe.
        polling_period (float): Time to wait for keyboard interrupts.
    """

    try:
        # Construct the policy
        policy = policy_constructor()

        # Policy setup complete
        pipe.send((None, True))

        while True:
            # Short block for keyboard interrupts
            if not pipe.poll(polling_period):
                continue
            command, data = pipe.recv()
            if command == "act":
                result = policy.act(data)
                pipe.send((result, True))
            elif command == "save":
                policy.save(data)
                pipe.send((None, True))
            elif command == "write_to_tb":
                policy.write_to_tb(data)
                pipe.send((None, True))
            elif command == "close":
                break
            else:
                raise KeyError(f"Received unknown command `{command}`.")
    except KeyboardInterrupt:
        error = (sys.exc_info()[0], "Traceback is hidden.")
        pipe.send((error, False))
    except Exception:
        error = sys.exc_info()[:2]
        pipe.send((error, False))
    finally:
        policy.close()
        pipe.close()
