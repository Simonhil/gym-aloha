from pathlib import Path

### Simulation envs fixed constants
DT = 0.02  # 0.02 ms -> 1/0.2 = 50 hz
FPS = 50


JOINTS = [
    # absolute joint position
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_angle",
    "left_arm_wrist_rotate",
    # normalized gripper position 0: close, 1: open
    "left_arm_gripper",
    # absolute joint position
    "right_arm_waist",
    "right_arm_shoulder",
    "right_arm_elbow",
    "right_arm_forearm_roll",
    "right_arm_wrist_angle",
    "right_arm_wrist_rotate",
    # normalized gripper position 0: close, 1: open
    "right_arm_gripper",
]

ACTIONS = [
    # position and quaternion for end effector
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_angle",
    "left_arm_wrist_rotate",
    # normalized gripper position (0: close, 1: open)
    "left_arm_gripper",
    "right_arm_waist",
    "right_arm_shoulder",
    "right_arm_elbow",
    "right_arm_forearm_roll",
    "right_arm_wrist_angle",
    "right_arm_wrist_rotate",
    # normalized gripper position (0: close, 1: open)
    "right_arm_gripper",
]


START_ARM_POSE = [0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,0.014,0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,0.024]

ASSETS_DIR = Path(__file__).parent.resolve() / "assets"  # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2
###########################Block Transfer###################
BLOCK_NAMES = [
    'box_1', 
    "rectangle_1",
    "rectangle_2",
    "rectangle_3"
]

###############put in box#############
BOX_CONTENSE_NAMES = [
        "cube"
    ]


############################Ball Maze###################
MAZE_FILES=[5,10,15,20,15,30]
NUMBER_BOARDS=4

selected_board=7
###########################Peg construction#####################
BODY_NAMES_PEG_CONSTRUCTION=[
    #"socket",
    "board",
    "peg"
]

######################join blocks##########

BODY_NAMES_JOIN_BLOCKS=[
  
    "hblock",
    "pblock",
]

############################ Helper functions ############################


def normalize_master_gripper_position(x):
    return (x - MASTER_GRIPPER_POSITION_CLOSE) / (
        MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
    )


def normalize_puppet_gripper_position(x):
    return (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
        PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
    )


def unnormalize_master_gripper_position(x):
    return x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE


def unnormalize_puppet_gripper_position(x):
    return x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE


def convert_position_from_master_to_puppet(x):
    return unnormalize_puppet_gripper_position(normalize_master_gripper_position(x))


def normalizer_master_gripper_joint(x):
    return (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)


def normalize_puppet_gripper_joint(x):
    return (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)


def unnormalize_master_gripper_joint(x):
    return x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE


def unnormalize_puppet_gripper_joint(x):
    return x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE


def convert_join_from_master_to_puppet(x):
    return unnormalize_puppet_gripper_joint(normalizer_master_gripper_joint(x))


def normalize_master_gripper_velocity(x):
    return x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)


def normalize_puppet_gripper_velocity(x):
    return x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)


def convert_master_from_position_to_joint(x):
    return (
        normalize_master_gripper_position(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
        + MASTER_GRIPPER_JOINT_CLOSE
    )


def convert_master_from_joint_to_position(x):
    return unnormalize_master_gripper_position(
        (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    )


def convert_puppet_from_position_to_join(x):
    return (
        normalize_puppet_gripper_position(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
        + PUPPET_GRIPPER_JOINT_CLOSE
    )


def convert_puppet_from_joint_to_position(x):
    return unnormalize_puppet_gripper_position(
        (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    )
