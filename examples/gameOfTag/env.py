import gym
import matplotlib.pyplot as plt
import numpy as np

from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
from smarts.env.wrappers import frame_stack as smarts_frame_stack
from typing import List


class SingleEnv(gym.Wrapper):
    def __init__(self, config):
        self.config = config
        self.agent_id = config["env_para"]["agent_ids"][0]

        # Agent interface
        agent_interface = smarts_agent_interface.AgentInterface(
            max_episode_steps=config["env_para"]["max_episode_steps"],
            neighborhood_vehicles=smarts_agent_interface.NeighborhoodVehicles(
                radius=config["env_para"]["neighborhood_radius"]
            ),
            rgb=smarts_agent_interface.RGB(
                width=256, height=256, resolution=config["env_para"]["rgb_wh"] / 256
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

        # Agent specs
        agent_specs = {
            self.agent_id: smarts_agent.AgentSpec(
                interface=agent_interface,
                agent_builder=None,
                observation_adapter=observation_adapter,
                reward_adapter=reward_adapter,
                action_adapter=action_adapter,
                info_adapter=info_adapter,
            )
        }

        # HiWayEnv
        env = smarts_hiway_env.HiWayEnv(
            scenarios=config["env_para"]["scenarios"],
            agent_specs=agent_specs,
            headless=config["env_para"]["headless"],
            visdom=config["env_para"]["visdom"],
            seed=config["env_para"]["seed"],
        )

        # Wrap env with FrameStack to stack multiple observations
        # env = smarts_frame_stack.FrameStack(env=env, num_stack=5, num_skip=4)

        # Initialize base env
        super(SingleEnv, self).__init__(env)

        # Action space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float
        )
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
        )

    def reset(self) -> np.ndarray:
        raw_states = self.env.reset()

        # Stack observation into 3D numpy matrix
        # states = {
        #     agent_id: stack_matrix(raw_state)
        #     for agent_id, raw_state in raw_states.items()
        # }
        states = raw_states

        return states[self.agent_id]

    def step(self, action):
        raw_states, rewards, dones, infos = self.env.step({self.agent_id: action})

        # Stack observation into 3D numpy matrix
        # states = {
        #     agent_id: stack_matrix(raw_state)
        #     for agent_id, raw_state in raw_states.items()
        # }
        states = raw_states
        
        # Plot for debugging purposes
        # import matplotlib.pyplot as plt
        # columns = 2 # number of frames stacked for each agent
        # rgb_gray = 3 # 3 for rgb and 1 for grayscale
        # n_states = len(raw_states.keys())
        # fig, axes = plt.subplots(1, n_states*columns, figsize=(10, 10))
        # fig.tight_layout()
        # ax = axes.ravel()
        # for row, (agent_id, state) in enumerate(states.items()):
        #     for col in range(columns):
        #         img = state[:,:,rgb_gray*col:rgb_gray*col+rgb_gray]
        #         ax[row*columns+col].imshow(img)
        #         ax[row*columns+col].set_title(agent_id)
        # plt.show()

        return (
            states[self.agent_id],
            rewards[self.agent_id],
            dones[self.agent_id],
            infos[self.agent_id],
        )

    def close(self):
        if self.env is not None:
            return self.env.close()
        return None


# def stack_matrix(states: List[np.ndarray]) -> np.ndarray:
#     # Stack 2D images along the depth dimension
#     print(states)
#     if states[0].ndim == 2 or states[0].ndim == 3:
#         return np.dstack(states)
#     else:
#         raise Exception(
#             f"Expected input numpy array with 2 or 3 dimensions, but received input with {states[0].ndim} dimensions."
#         )


def info_adapter(obs, reward, info):
    return info


def action_adapter(model_action):
    # Action space
    # throttle: [0, 1]
    # brake: [0, 1]
    # steering: [-1, 1]

    throttle, brake, steering = model_action
    # Modify action space limits
    throttle = (throttle + 1) / 2
    brake = (brake + 1) / 2
    return np.array([throttle, brake, steering], dtype=np.float)


def observation_adapter(obs) -> np.ndarray:
    # RGB grid map
    rgb = obs.top_down_rgb.data

    # Replace self color to yellow
    coloured_self = rgb.copy()
    # coloured_self[123:132, 126:130, 0] = 255
    # coloured_self[123:132, 126:130, 1] = 190
    # coloured_self[123:132, 126:130, 2] = 40

    # Convert rgb to grayscale image
    # grayscale = rgb2gray(coloured_self)

    # Center frames
    # frame = grayscale * 2 - 1
    frame = coloured_self.astype(np.uint8)

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
        reward -= 30
        print(f"Vehicle {ego.id} went off road.")
        return np.float(reward)

    # Reward for colliding
    for c in obs.events.collisions:
        reward -= 30
        print(f"Vehicle {ego.id} collided with vehicle {c.collidee_id}.")
        return np.float(reward)

    # Reward for staying on track
    reward += 1

    return np.float(reward)
