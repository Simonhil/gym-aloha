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
    kwargs={"obs_type": "pixels", "task": "transfer_cube", "task_description":"pick up red cube and transfer it from right to left "},
)

register(
    id="gym_aloha/AlohaTransferCube-v1",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    #kwargs={"obs_type": "pixels_agent_pos", "task": "transfer_cube"},
    kwargs={"obs_type": "pixels_agent_pos", "task": "pick up red cube and transfer it from right to left ", "task_description":"pick up red cube and transfer it from right to left "},
)



register(
    id="gym_aloha/AlohaTest-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task": "pick up red cube and transfer it from right to left ", "task_description":"pick up red cube and transfer it from right to left "},
)

register(
    id="gym_aloha/AlohaBockStacking-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task": "place the yellow block on the red one and stack put the other two simultaniously on the yellow block ", "task_description":"place the yellow block on the red one and stack put the other two simultaniously on the yellow block "},
)

register(
    id="gym_aloha/AlohaBallMaze-v0",
    entry_point="gym_aloha.env:AlohaMazeEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task": "move the board so that the blue ball hits the pink hexagon", "task_description":"move the board so that the blue ball hits the pink hexagon"},
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
    kwargs={"obs_type": "pixels_agent_pos", "task":"join the two blocks and hang them on the wall" , "task_description": "join the two blocks and hang them on the wall"},
)

register(
    id="gym_aloha/PutInBox-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task":"pick up the block, open the box and place the blcok into the box", "task_description":"pick up the block, open the box and place the blcok into the box" },
)