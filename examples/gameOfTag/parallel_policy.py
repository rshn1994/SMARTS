# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import multiprocessing as mp
import numpy as np
import sys
import warnings

from ppo import RL, PPO
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union


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
        """The policies can be different but must use the same input and output specs.
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

        mp_ctx = mp.get_context()
        self.policy_constructors = policy_constructors

        self.error_queue = mp_ctx.Queue()
        self.parent_pipes = []
        self.processes = []
        for idx, env_constructor in enumerate(self.env_constructors):
            parent_pipe, child_pipe = mp_ctx.Pipe()
            process = mp_ctx.Process(
                target=_worker,
                name=f"Worker<{type(self).__name__}>-<{idx}>",
                args=(
                    idx,
                    sim_name,
                    CloudpickleWrapper(env_constructor),
                    child_pipe,
                    self.error_queue,
                    self._polling_period,
                ),
            )
            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)

            # Daemonic subprocesses quit when parent process quits. However, daemonic
            # processes cannot spawn children. Hence, `process.daemon` is set to False.
            process.daemon = False
            process.start()
            child_pipe.close()

    #     # Wait for all environments to successfully startup
    #     _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
    #     self._raise_if_errors(successes)

    #     # Get and check observation and action spaces
    #     observation_space, action_space = self._get_spaces()

    def save(self):
        pass

    def act(self):
        pass

    def write_to_tb(self):
        pass

    def model(self):
        pass

    def optimizer(self):
        pass


    #     for pipe, seed in zip(self.parent_pipes, seeds):
    #         pipe.send(("seed", seed))
    #     seeds, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
    #     self._raise_if_errors(successes)

    #     return seeds

    # def reset_wait(
    #     self, timeout: Union[int, float, None] = None
    # ) -> Sequence[Dict[str, Any]]:
    #     """Waits for all environments to reset.
    #     Args:
    #         timeout (Union[int, float, None], optional): Seconds to wait before timing out.
    #             Defaults to None, and never times out.
    #     Raises:
    #         NoAsyncCallError: If `reset_wait` is called without calling `reset_async`.
    #         mp.TimeoutError: If response is not received from pipe within `timeout` seconds.
    #     Returns:
    #         Sequence[Dict[str, Any]]: A batch of observations from the vectorized environment.
    #     """

    #     self._assert_is_running()
    #     if self._state != AsyncState.WAITING_RESET:
    #         raise NoAsyncCallError(
    #             "Calling `reset_wait` without any prior call to `reset_async`.",
    #             AsyncState.WAITING_RESET.value,
    #         )

    #     if not self._poll(timeout):
    #         self._state = AsyncState.DEFAULT
    #         raise mp.TimeoutError(
    #             "The call to `reset_wait` has timed out after "
    #             "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
    #         )

    #     observations, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
    #     self._raise_if_errors(successes)
    #     self._state = AsyncState.DEFAULT

    #     return observations

    # def step_wait(
    #     self, timeout: Union[int, float, None] = None
    # ) -> Tuple[
    #     Sequence[Dict[str, Any]],
    #     Sequence[Dict[str, float]],
    #     Sequence[Dict[str, bool]],
    #     Sequence[Dict[str, Any]],
    # ]:
    #     """Waits and returns batched (observations, rewards, dones, infos) from all environments after a single step.
    #     Args:
    #         timeout (Union[int, float, None], optional): Seconds to wait before timing out.
    #             Defaults to None, and never times out.
    #     Raises:
    #         NoAsyncCallError: If `step_wait` is called without calling `step_async`.
    #         mp.TimeoutError: If data is not received from pipe within `timeout` seconds.
    #     Returns:
    #         Tuple[ Sequence[Dict[str, Any]], Sequence[Dict[str, float]], Sequence[Dict[str, bool]], Sequence[Dict[str, Any]] ]:
    #             Returns (observations, rewards, dones, infos). Each tuple element is a batch from the vectorized environment.
    #     """

    #     self._assert_is_running()
    #     if self._state != AsyncState.WAITING_STEP:
    #         raise NoAsyncCallError(
    #             "Calling `step_wait` without any prior call to `step_async`.",
    #             AsyncState.WAITING_STEP.value,
    #         )

    #     if not self._poll(timeout):
    #         self._state = AsyncState.DEFAULT
    #         raise mp.TimeoutError(
    #             "The call to `step_wait` has timed out after "
    #             "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
    #         )

    #     results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
    #     self._raise_if_errors(successes)
    #     self._state = AsyncState.DEFAULT
    #     observations_list, rewards, dones, infos = zip(*results)

    #     return (
    #         observations_list,
    #         rewards,
    #         dones,
    #         infos,
    #     )

    # def _get_spaces(self) -> Tuple[gym.Space, gym.Space]:
    #     for pipe in self.parent_pipes:
    #         pipe.send(("_get_spaces", None))
    #     spaces, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
    #     self._raise_if_errors(successes)

    #     observation_space = spaces[0][0]
    #     action_space = spaces[0][1]

    #     if not all([space[0] == observation_space for space in spaces]) or not all(
    #         [space[1] == action_space for space in spaces]
    #     ):
    #         raise RuntimeError(
    #             f"Expected all environments to have the same observation and action"
    #             f"spaces, but got {spaces}."
    #         )

    #     return observation_space, action_space


# def _worker(
#     index: int,
#     sim_name: str,
#     env_constructor: CloudpickleWrapper,
#     auto_reset: bool,
#     pipe: mp.connection.Connection,
#     error_queue: mp.Queue,
#     polling_period: float = 0.1,
# ):
#     """Process to build and run an environment. Using a pipe to
#     communicate with parent, the process receives action, steps
#     the environment, and returns the observations.
#     Args:
#         index (int): Environment index number.
#         env_constructor (CloudpickleWrapper): Callable which constructs the environment.
#         auto_reset (bool): If True, auto resets environment when episode ends.
#         pipe (mp.connection.Connection): Child's end of the pipe.
#         error_queue (mp.Queue): Queue to communicate error messages.
#         polling_period (float): Time to wait for keyboard interrupts.
#     """

#     # Name and construct the environment
#     name = f"env_{index}"
#     if sim_name:
#         name = sim_name + "_" + name
#     env = env_constructor(sim_name=name)

#     # Environment setup complete
#     pipe.send((None, True))

#     try:
#         while True:
#             # Short block for keyboard interrupts
#             if not pipe.poll(polling_period):
#                 continue
#             command, data = pipe.recv()
#             if command == "reset":
#                 observation = env.reset()
#                 pipe.send((observation, True))
#             elif command == "step":
#                 observation, reward, done, info = env.step(data)
#                 if done["__all__"] and auto_reset:
#                     # Actual final observations can be obtained from `info`:
#                     # ```
#                     # final_obs = info[agent_id]["env_obs"]
#                     # ```
#                     observation = env.reset()
#                 pipe.send(((observation, reward, done, info), True))
#             elif command == "seed":
#                 env_seed = env.seed(data)
#                 pipe.send((env_seed, True))
#             elif command == "close":
#                 pipe.send((None, True))
#                 break
#             elif command == "_get_spaces":
#                 pipe.send(((env.observation_space, env.action_space), True))
#             else:
#                 raise KeyError(f"Received unknown command `{command}`.")
#     except KeyboardInterrupt:
#         error_queue.put((index, sys.exc_info()[0], "Traceback is hidden."))
#         pipe.send((None, False))
#     except Exception:
#         error_queue.put((index,) + sys.exc_info()[:2])
#         pipe.send((None, False))
#     finally:
#         env.close()
#         pipe.close()
