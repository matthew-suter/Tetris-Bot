import gymnasium as gym

env = gym.make("ALE/Tetris-v5", render_mode="human", obs_type="grayscale")
observation, info = env.reset()


def greyscale_to_one_hot(greyscale):
    result =  greyscale[28:204:8, 24:64:4] # Trim to only include the playspace, and then downscale
    result = (result != 0x6F)*1 # Check if background colour (Hex code 0x6F), and convert to numerical boolean using disgusting typecasting
    return result




for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)


    # trimmed_obs = observation[27:203, 22:64] # Trim down the playspace
    # trimmed_obs = observation[27:203, 23:63] # Trim down the playspace, making it the right grid size

    # trimmed_obs = observation[28:204:8, 24:64:4] # Trim, shift to avoid the inter-cell gaps, and downscale

    trimmed_obs = greyscale_to_one_hot(observation)

    for row in range(trimmed_obs.shape[0]):
        for col in range(trimmed_obs.shape[1]):
            # print(charmap(trimmed_obs[row, col]), end="")
            print(trimmed_obs[row, col], end="")
        print()
        # print(row, observation[row].shape)
        # print(observation[row])
    # print(observation)

    # input()
    print()

    if terminated or truncated:
        observation, info = env.reset()

env.close()