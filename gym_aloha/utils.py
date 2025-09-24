import numpy as np





def sample_box_pose(seed=None):
    x_range = [-0.1, 0.1]
    y_range = [-0.1, 0.1]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose(seed=None):
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


def sample_coordinates_max(ranges, y_max, x_max, seed=None):
    cube_position = []
    rng = np.random.RandomState(seed)
    while True:
        cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        if not (cube_position[0] > x_max  and  cube_position[1] > y_max):
            break
    print("cube_position: " + str(cube_position))
    return cube_position

def sample_coordinates_mid(ranges, y_min, y_max, x_min, x_max, seed=None):
    cube_position = []
    rng = np.random.RandomState(seed)
    while True:
        cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        if not (cube_position[0] > x_max  or cube_position[0]< x_min )  and  not (cube_position[1] > y_max or cube_position[1]< y_min):
            break
    print("cube_position: " + str(cube_position))
    return cube_position
            
    
def sample_put_in_box_pose(seed=None):
    x_range = [-0.1, 0.1]
    y_range = [-0.1, 0.1]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = sample_coordinates_max(ranges, -0.07, -0.07, seed)

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_block_stacking(seed=None):
    x_range = [-0.1, 0.1]
    y_range = [-0.1, 0.1]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    to_return=[]
    cube_quat = np.array([1, 0, 0, 0])
    base_position = sample_coordinates_mid(ranges, -0.06, 0.04, -0.06,0.06 , seed)
    to_return.append(np.concatenate([base_position, cube_quat]))
    beam_position = sample_coordinates_mid(ranges, -0.1, -0.06, -0.1, 0.1, seed)
    to_return.append(np.concatenate([beam_position, cube_quat]))
    block1_position = sample_coordinates_mid(ranges, 0.06, 0.1, -0.1, -0.01, seed)
    to_return.append(np.concatenate([block1_position, cube_quat]))
    block2_position= sample_coordinates_mid(ranges, 0.06, 0.1, 0, 0.1, seed)
    to_return.append(np.concatenate([block2_position, cube_quat]))
    print("return" + str(to_return))
    return to_return