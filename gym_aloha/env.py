import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces
from mujoco import viewer as mj_viewer
from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask
from gym_aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose


class AlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self._env = self._make_env_task(self.task)

        if self.obs_type == "state":
            raise NotImplementedError()
            self.observation_space = spaces.Box(
                low=np.array([0] * len(JOINTS)),  # ???
                high=np.array([255] * len(JOINTS)),  # ???
                dtype=np.float64,
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "overhead_cam": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist_cam_left": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist_cam_right": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                             "overhead_cam": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist_cam_left": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist_cam_right": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)

        model = self._env.physics.model.ptr
        data = self._env.physics.data.ptr

        self.viewer = mj_viewer.launch_passive(
        model=model,
        data=data,
    )




    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        image = { "overhead_cam":self._env.physics.render(height=height, width=width, camera_id="overhead_cam"),
                 "wrist_cam_left":self._env.physics.render(height=height, width=width, camera_id="wrist_cam_left"),
                 "wrist_cam_right":self._env.physics.render(height=height, width=width, camera_id="wrist_cam_right"),
        }
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif task_name == "insertion":
            xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "end_effector_transfer_cube":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif task_name == "end_effector_insertion":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {"overhead_cam": raw_obs["images"]["overhead_cam"].copy(), "wrist_cam_left": raw_obs["images"]["wrist_cam_left"].copy()
                   ,"wrist_cam_right": raw_obs["images"]["wrist_cam_right"].copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {"overhead_cam": raw_obs["images"]["overhead_cam"].copy(), "wrist_cam_left": raw_obs["images"]["wrist_cam_left"].copy()
                   ,"wrist_cam_right": raw_obs["images"]["wrist_cam_right"].copy()},
                "agent_pos": raw_obs["qpos"],
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        self.viewer.sync()
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        _, reward, _, raw_obs = self._env.step(action)
        self.viewer.sync()

        # TODO(rcadene): add an enum
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        pass