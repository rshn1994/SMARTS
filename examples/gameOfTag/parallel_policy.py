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
import sys
import warnings

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
        mp_ctx = mp.get_context()

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

    def act(self, states: Dict[str, Any]):
        #     for name, value in dic.items():
        #         self.parent_pipes[policy_id].send((method,val))

        #     payload = args, kwargs
        #     self._conn.send((name, payload))
        #     return self._receive
        # results = {policy_id: pipe.recv() for policy_id, pipe in self.parent_pipes.items()}
        # self._raise_if_errors(results)

        #     for pipe, seed in zip(self.parent_pipes, seeds):
        #         pipe.send(("seed", seed))
        #     seeds, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        #     self._raise_if_errors(successes)

        return 1

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

        for policy_id, (error, _) in results.items():
            if error:
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
