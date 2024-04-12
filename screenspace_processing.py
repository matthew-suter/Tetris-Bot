import gymnasium as gym

env = gym.make("ALE/Tetris-v5", render_mode="human", obs_type="grayscale")
observation, info = env.reset()


def greyscale_to_one_hot(greyscale):
    result =  greyscale[28:204:8, 24:64:4] # Trim to only include the playspace, shifting one pixel down and left to avoid inter-block gaps, and then downscale
    result = (result != 0x6F)*1 # Check if background colour (Hex code 0x6F), and convert to numerical boolean using disgusting typecasting
    return result




for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    trimmed_obs = greyscale_to_one_hot(observation)

    for row in range(trimmed_obs.shape[0]):
        for col in range(trimmed_obs.shape[1]):
            print(trimmed_obs[row, col], end="")
        print()

    # input() # Enter to advance timesteps
    print()

    if terminated or truncated:
        observation, info = env.reset()

env.close()