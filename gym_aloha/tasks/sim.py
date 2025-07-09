import collections

import numpy as np
from dm_control.suite import base
from gym_aloha.constants import (
    BLOCK_NAMES,
    BODY_NAMES_JOIN_BLOCKS,
    BODY_NAMES_PEG_CONSTRUCTION,
    BOX_CONTENSE_NAMES,
    START_ARM_POSE,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
    unnormalize_puppet_gripper_position,
)
import gym_aloha.constants

BOX_POSE = []  # to be changed from outside

"""
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (6),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (6),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7 : 7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = unnormalize_puppet_gripper_position(normalized_left_gripper_action)
        right_gripper_action = unnormalize_puppet_gripper_position(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action]
        full_right_gripper_action = [right_gripper_action]

        env_action = np.concatenate(
            [left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action]
        )
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [normalize_puppet_gripper_position(left_qpos_raw[6])]
        right_gripper_qpos = [normalize_puppet_gripper_position(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [normalize_puppet_gripper_velocity(left_qvel_raw[6])]
        right_gripper_qvel = [normalize_puppet_gripper_velocity(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["overhead_cam"] = physics.render(height=224, width=224, camera_id="overhead_cam")
        obs["images"]["wrist_cam_left"] = physics.render(height=224, width=224, camera_id="wrist_cam_left")
        obs["images"]["wrist_cam_right"] = physics.render(height=224, width=224, camera_id="wrist_cam_right")

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            #physics.named.data.qpos[:14] = START_ARM_POSE
            physics.named.data.qpos[:7] = START_ARM_POSE[:7]
            physics.named.data.qpos[8:15] = START_ARM_POSE[7:]
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] =  BOX_POSE.pop(0)
            BOX_POSE.pop
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        touch_left_gripper = ("red_box", "left_vx300s_8_custom_finger_right") in all_contact_pairs
        touch_right_gripper = ("red_box", "right_vx300s_8_custom_finger_left") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward

class BallMaze(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            #physics.named.data.qpos[:14] = START_ARM_POSE
            physics.named.data.qpos[:7] = START_ARM_POSE[:7]
            physics.named.data.qpos[8:15] = START_ARM_POSE[7:]
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = [0, 0 , 0.05, 0 ,0 ,0 ,0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        selected_board = gym_aloha.constants.selected_board
        marker_name=f"marker{selected_board}"
        marker_id = physics.model.name2id(marker_name, 'geom')
        marker_pos = physics.named.data.geom_xpos[marker_id][:2]

        ball_name="blue_ball"
        ball_id=physics.model.name2id(ball_name, 'geom')
        ball_pos=physics.named.data.geom_xpos[ball_id][:2]
    
        diff = abs(ball_pos - marker_pos)
        reward = 0
        if diff[0] <0.002 and diff [1] <0.002:
            reward = 4
        else:reward = 0
        return reward

class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:14] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2 :] = BOX_POSE[0]  # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if (
            touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table)
        ):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward

class BlockStackingTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.num_blocks = 0

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            #physics.named.data.qpos[:14] = START_ARM_POSE
            physics.named.data.qpos[:7] = START_ARM_POSE[:7]
            physics.named.data.qpos[8:15] = START_ARM_POSE[7:]
            np.copyto(physics.data.ctrl, START_ARM_POSE)

            block_ids = []
            for name in BLOCK_NAMES:
                   block_ids.append(physics.model.name2id(name, 'body'))
            print(BOX_POSE)
            for i in range(len(block_ids)):
                assert len(BOX_POSE) != 0
                joint_id = physics.model.body_jntadr[block_ids[i]]
                qpos_addr = physics.model.jnt_qposadr[joint_id]
                physics.data.qpos[qpos_addr:qpos_addr+3] = BOX_POSE[0][:3]
                BOX_POSE.pop(0)
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        #TODO be implemented once needed
        
        return 0

        
class PegConstructionTask(BimanualViperXTask):
        def __init__(self, random=None):
            super().__init__(random=random)
            self.max_reward = 4
            self.num_blocks = 0

        def initialize_episode(self, physics):
            """Sets the state of the environment at the start of each episode."""
            # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
            # reset qpos, control and box position
            with physics.reset_context():
                #physics.named.data.qpos[:14] = START_ARM_POSE
                physics.named.data.qpos[:7] = START_ARM_POSE[:7]
                physics.named.data.qpos[8:15] = START_ARM_POSE[7:]
                np.copyto(physics.data.ctrl, START_ARM_POSE)

                piece_ids = []
                for name in BODY_NAMES_PEG_CONSTRUCTION:
                    piece_ids.append(physics.model.name2id(name, 'body'))

                for i in range(len(piece_ids)):
                    assert len(BOX_POSE) != 0
                    print(BOX_POSE[0])
                    joint_id = physics.model.body_jntadr[piece_ids[i]]
                    qpos_addr = physics.model.jnt_qposadr[joint_id]
                    physics.data.qpos[qpos_addr:qpos_addr+3] = BOX_POSE[0][:3]
                    BOX_POSE.pop(0)
                # print(f"{BOX_POSE=}")
            super().initialize_episode(physics)

        @staticmethod
        def get_env_state(physics):
            env_state = physics.data.qpos.copy()[16:]
            return env_state

        def get_reward(self, physics):
            #TODO be implemented once needed
            
            return 0

class JoinBlocksTask(BimanualViperXTask):
        def __init__(self, random=None):
            super().__init__(random=random)
            self.max_reward = 4
            self.num_blocks = 0

        def initialize_episode(self, physics):
            """Sets the state of the environment at the start of each episode."""
            # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
            # reset qpos, control and box position
            with physics.reset_context():
                #physics.named.data.qpos[:14] = START_ARM_POSE
                physics.named.data.qpos[:7] = START_ARM_POSE[:7]
                physics.named.data.qpos[8:15] = START_ARM_POSE[7:]
                np.copyto(physics.data.ctrl, START_ARM_POSE)

                piece_ids = []
                for name in BODY_NAMES_JOIN_BLOCKS:
                    piece_ids.append(physics.model.name2id(name, 'body'))

                for i in range(len(piece_ids)):
                    assert len(BOX_POSE) != 0
                    joint_id = physics.model.body_jntadr[piece_ids[i]]
                    qpos_addr = physics.model.jnt_qposadr[joint_id]
                    physics.data.qpos[qpos_addr:qpos_addr+3] = BOX_POSE[0][:3]
                    BOX_POSE.pop(0)
                # print(f"{BOX_POSE=}")
            super().initialize_episode(physics)

        @staticmethod
        def get_env_state(physics):
            env_state = physics.data.qpos.copy()[16:]
            return env_state

        def get_reward(self, physics):
            selected_board = gym_aloha.constants.selected_board
            block_peg_name=f"block_peg"
            block_peg_id = physics.model.name2id(block_peg_name, 'geom')
            block_peg_pos = physics.named.data.geom_xpos[block_peg_id]
          
            wall_peg_name=f"wall_peg"
            wall_peg_id = physics.model.name2id(wall_peg_name, 'geom')
            wall_peg_pos = physics.named.data.geom_xpos[wall_peg_id]

            wall_marker_name=f"marker_wall"
            wall_marker_id = physics.model.name2id(wall_marker_name, 'geom')
            wall_marker_pos = physics.named.data.geom_xpos[wall_marker_id]

            block_marker_name=f"marker_block"
            block_marker_id = physics.model.name2id(block_marker_name, 'geom')
            block_marker_pos = physics.named.data.geom_xpos[block_marker_id]





            diff_wall= abs(wall_peg_pos - wall_marker_pos)
            reward_wall= 0
            if diff_wall[0] <0.005 and diff_wall [1] <0.005 and diff_wall[2]<0.07:
                reward_wall = 2
            else:
                reward_wall = 0

            
            diff_block = abs(block_peg_pos - block_marker_pos)

            reward_block = 0
            if diff_block[0] < 0.003 and diff_block[1] < 0.003 and diff_block[2] < 0.05:
                reward_block = 2
            else:
                reward_block = 0

            
            return reward_wall + reward_block
            
        

class PutInBoxTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.num_blocks = 0

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            #physics.named.data.qpos[:14] = START_ARM_POSE
            physics.named.data.qpos[:7] = START_ARM_POSE[:7]
            physics.named.data.qpos[8:15] = START_ARM_POSE[7:]
            np.copyto(physics.data.ctrl, START_ARM_POSE)

            block_ids = []
            for name in BOX_CONTENSE_NAMES:
                   block_ids.append(physics.model.name2id(name, 'body'))

            for i in range(len(block_ids)):
                assert len(BOX_POSE) != 0
                joint_id = physics.model.body_jntadr[block_ids[i]]
                qpos_addr = physics.model.jnt_qposadr[joint_id]
                physics.data.qpos[qpos_addr:qpos_addr+3] = BOX_POSE[0][:3]
                BOX_POSE.pop(0)
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        touch_left_gripper = ("red_box", "left_vx300s_8_custom_finger_right") in all_contact_pairs
        touch_right_gripper=any(
        "right_vx300s_8_custom_finger_left" in pair for pair in all_contact_pairs
        )
        box_closed=("lid_front", "marker") in all_contact_pairs
        in_box=("red_box", "marker") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs


        # reward=0
        # if in_box and box_closed:
        #     reward=4
        # else:
        #     reward = 0

        marker_name=f"marker"
        marker_id = physics.model.name2id(marker_name, 'geom')
        marker_pos = physics.named.data.geom_xpos[marker_id]
        
        red_block_name=f"red_box"
        red_block_id = physics.model.name2id(red_block_name, 'geom')
        red_block_pos = physics.named.data.geom_xpos[red_block_id]
       
        diff_red_block_in= abs(marker_pos - red_block_pos)
        print(diff_red_block_in)
        reward= 0
        if diff_red_block_in[0] <0.08 and diff_red_block_in [1] <0.08 and diff_red_block_in[2]<0.3 and not touch_right_gripper:
            reward= 4
        print(reward)
        return reward