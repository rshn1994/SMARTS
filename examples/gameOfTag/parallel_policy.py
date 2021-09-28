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
from smarts.env.utils.cloud_pickle import CloudpickleWrapper
from typing import Any, Callable, Dict, List, Sequence, Text, Tuple, Union


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

        self.closed=False
        mp_ctx = mp.get_context()

        self.error_queue = mp_ctx.Queue()
        self.parent_pipes = {}
        self.processes = {}
        for idx, (policy_id, policy_constructor) in enumerate(policy_constructors.items()):
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
        results = {policy_id: pipe.recv() for policy_id, pipe in self.parent_pipes.items()}
        self._raise_if_errors(results)


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

    def close(self, terminate=True):
        """Closes all processes alive.

        Args:
            terminate (bool, optional): If `True`, then the `close` operation is forced and all
                processes are terminated. Defaults to True.
        """
        if terminate:
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes.values():
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes.values():
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes.values():
            if pipe is not None:
                pipe.close()
        for process in self.processes.values():
            process.join()

    def __del__(self):
        if not self.closed:
            self.close()
            self.closed=True

    #     observations, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])


def _worker(
    policy_constructor: CloudpickleWrapper,
    pipe: mp.connection.Connection,
    polling_period: float = 0.1,
):
    """Process to build and run a policy. Using a pipe to communicate with parent, the
    process receives instructions, and returns results.
    Args:
        index (int): Policy index number.
        policy_id (str): Policy id.
        policy_constructor (CloudpickleWrapper): Callable which constructs the policy.
        pipe (mp.connection.Connection): Child's end of the pipe.
        error_queue (mp.Queue): Queue to communicate error messages.
        polling_period (float): Time to wait for keyboard interrupts.
    """

    # Construct the environment
    policy = policy_constructor()

    # Environment setup complete
    pipe.send((None, True))

    try:
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
                pipe.send((None, True))
                break
            else:
                raise KeyError(f"Received unknown command `{command}`.")
    except (KeyboardInterrupt, EOFError):
        error = (sys.exc_info()[0], "Traceback is hidden.")
        pipe.send((error, False))
    except Exception:
        error = (sys.exc_info()[:2])
        pipe.send((error, False))
    finally:
        policy.close()
        pipe.close()
