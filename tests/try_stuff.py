import time
import cv2
import imageio
import gymnasium as gym
from dm_control import mujoco
import numpy as np
import trimesh
import gym_aloha
from gym_aloha.constants import ASSETS_DIR, selected_board
from mujoco import viewer as mj_viewer


env = gym.make("gym_aloha/AlohaBallMaze-v0")
observation, info = env.reset()
frames = []
for i in range(1000):
    action = env.action_space.sample()
    #action = np.array([0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,-0.25,0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,0.014])
    #print(len(action))
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image["wrist_cam_right"])
    combined = np.concatenate(list(image.values()), axis=1)

    # Show with OpenCV
    scale_factor = 0.75
    combined = np.concatenate(list(image.values()), axis=1)
    resized = cv2.resize(combined, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Multi-Camera Views", resized [..., ::-1])  # RGB to BGR

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if terminated or truncated:
        observation, info = env.reset()
    # # env.reset()
    # env.reset()
    # print("test")
    # time.sleep(4)

env.close()
cv2.destroyAllWindows()
# imageio.mimsave("example.mp4", np.stack(frames), fps=25)
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


# mesh_or_scene = trimesh.load('/home/i53/student/shilber/3dstuff/v-hacd/app/build/decomp.stl')

# if isinstance(mesh_or_scene, trimesh.Scene):
#     # Combine all geometries into a single mesh
#     mesh = trimesh.util.concatenate(mesh_or_scene.dump())
# else:
#      mesh = mesh_or_scene
# print("Number of faces:", mesh.faces.shape[0])
# mesh.export('/home/i53/student/shilber/3dstuff/v-hacd/app/build/decomp_output_binary.stl', file_type='stl')
