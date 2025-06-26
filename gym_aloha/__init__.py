from gymnasium.envs.registration import register

register(
    id="gym_aloha/AlohaInsertion-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task": "insertion"},
)

register(
    id="gym_aloha/AlohaTransferCube-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    #kwargs={"obs_type": "pixels_agent_pos", "task": "transfer_cube"},
    kwargs={"obs_type": "pixels", "task": "transfer_cube"},
)

register(
    id="gym_aloha/AlohaTransferCube-v1",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    #kwargs={"obs_type": "pixels_agent_pos", "task": "transfer_cube"},
    kwargs={"obs_type": "pixels_agent_pos", "task": "transfer_cube"},
)



register(
    id="gym_aloha/AlohaTest-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task": "test"},
)

register(
    id="gym_aloha/AlohaBockStacking-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task": "block_stacking"},
)

register(
    id="gym_aloha/AlohaBallMaze-v0",
    entry_point="gym_aloha.env:AlohaMazeEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task": "ball_maze"},
)
register(
    id="gym_aloha/AlohaPegConstruction-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task":"peg_construction"},
)

register(
    id="gym_aloha/AlohaJoinBlocks-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task":"join_blocks"},
)

register(
    id="gym_aloha/PutInBox-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task":"put_in_box"},
)