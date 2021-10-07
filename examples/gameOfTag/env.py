import gym
import matplotlib.pyplot as plt
import numpy as np
import time

from examples.gameOfTag import agent as got_agent
from skimage.color import rgb2gray
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
from smarts.env.wrappers import frame_stack as smarts_frame_stack
from typing import Dict, List
from smarts.core.agent import Agent
from smarts.core.sensors import Observation

class SingleAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.edge_id != obs.via_data.near_via_points[0].edge_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )



class SingleEnv(gym.Wrapper):
    def __init__(self, config):
        self.config = config
        self.neighborhood_radius = config["env_para"]["neighborhood_radius"]
        self.rgb_wh = config["env_para"]["rgb_wh"]

        agent_interface = smarts_agent_interface.AgentInterface(
            max_episode_steps=config["env_para"]["max_episode_steps"],
            neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
                radius=self.neighborhood_radius
            ),
            rgb=smarts_agent_interface.RGB(
                width=256, height=256, resolution=self.rgb_wh / 256
            ),
            vehicle_color="Blue",
            action=smarts_controllers.ActionSpaceType.Continuous,
            done_criteria=smarts_agent_interface.DoneCriteria(
                collision=True,
                off_road=True,
                off_route=False,
                on_shoulder=False,
                wrong_way=False,
                not_moving=False,
            ),
        )

        agent_specs = {
            "Agent_001": smarts_agent.AgentSpec(
                interface=agent_interface,
                agent_builder=None,
                observation_adapter=observation_adapter,
                reward_adapter=reward_adapter,
                action_adapter=action_adapter,
                info_adapter=info_adapter,
            )
        }

        env = smarts_hiway_env.HiWayEnv(
            scenarios=config["env_para"]["scenarios"],
            agent_specs=agent_specs,
            headless=config["env_para"]["headless"],
            visdom=config["env_para"]["visdom"],
            seed=42,
        )

        super(SingleEnv, self).__init__(env)


        # Wrap env with FrameStack to stack multiple observations
        self.env = smarts_frame_stack.FrameStack(env=env, num_stack=9, num_skip=4)

        # Set action space and observation space
        self.action_space = gym.spaces.Box(
            np.array([-1, -1, -1]), np.array([+1, +1, +1]), dtype=np.float32
        )  # throttle, break, steering
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(256, 256, 3), dtype=np.float32
        )

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment, if done is true, must clear obs array.

        :return: the observation of gym environment
        """

        raw_states = self.env.reset()

        # Stack observation into 3D numpy matrix
        states = {
            agent_id: stack_matrix(raw_state)
            for agent_id, raw_state in raw_states.items()
        }

        self.init_state = states
        return states

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        Accepts an action and returns a tuple (state, reward, done, info).

        :param action: action
        :param agent_index: the index of agent
        :return: state, reward, done, info
        """

        raw_states, rewards, dones, infos = self.env.step(action)

        # Stack observation into 3D numpy matrix
        states = {
            agent_id: stack_matrix(raw_state)
            for agent_id, raw_state in raw_states.items()
        }

        # Plot for debugging purposes
        # import matplotlib.pyplot as plt
        # fig=plt.figure(figsize=(10,10))
        # columns = 3
        # rows = len(states.keys())
        # for row, (agent_id, state) in enumerate(states.items()):
        #     for col in range(0, columns):
        #         img = state[:,:,col]
        #         fig.add_subplot(rows, columns, row*columns + col + 1)
        #         plt.imshow(img)
        # plt.show()

        return states, rewards, dones, infos

    def close(self):
        if self.env is not None:
            return self.env.close()
        return None


def info_adapter(obs, reward, info):
    return reward


def action_adapter(model_action):
    # Action space
    # throttle: [0, 1]
    # brake: [0, 1]
    # steering: [-1, 1]

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

def reward_adapter(obs, env_reward):
    reward = 0
    ego = obs.ego_vehicle_state

    # Penalty for driving off road
    if obs.events.off_road:
        reward -= 10
        return np.float32(reward)

     # Reward for colliding
    for c in obs.events.collisions:
        reward -= 10
        print(f"Vehicle {ego.id} collided with vehicle {c.collidee_id}.")
        return np.float32(reward)

    reward += 1

    return np.float32(reward)
