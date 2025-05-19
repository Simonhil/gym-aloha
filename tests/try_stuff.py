import time
import imageio
import gymnasium as gym
from dm_control import mujoco
import numpy as np
import gym_aloha
from gym_aloha.constants import ASSETS_DIR
from mujoco import viewer as mj_viewer

env = gym.make("gym_aloha/AlohaTest-v0")
observation, info = env.reset()
frames = []

for _ in range(1000):
    action = env.action_space.sample()
    #action = np.array([0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,0.024 ,0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,0.024])
    #print(len(action))
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation['agent_pos'])
    image = env.render()
    frames.append(image["wrist_cam_right"])

    if terminated or truncated:
        observation, info = env.reset()
    # env.reset()
    # time.sleep(1)

env.close()
#imageio.mimsave("example.mp4", np.stack(frames), fps=25)
# xml_path = ASSETS_DIR / "bimanual_viperx_ball_maze.xml"
# physics = mujoco.Physics.from_xml_path(str(xml_path))
# model = physics.model.ptr
# data = physics.data.ptr

# viewer = mj_viewer.launch_passive(
#     model=model,
#     data=data,
# )
# for i in range(1000):
#      viewer.sync()
#      time.sleep(0.02)