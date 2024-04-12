import gymnasium as gym
import additional_reward

env = gym.make("ALE/Tetris-v5", render_mode="human", obs_type="grayscale")
observation, info = env.reset()


def greyscale_to_one_hot(greyscale):
    grid =  greyscale[28:204:8, 24:64:4] # Trim to only include the playspace, shifting one pixel down and left to avoid inter-block gaps, and then downscale
    grid = (grid != 0x6F)*1 # Check if background colour (Hex code 0x6F), and convert to numerical boolean using disgusting typecasting
    return grid




# def scored_this_turn()


for i in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if i != 0:
        last_trimmed_obs = trimmed_obs
    trimmed_obs = greyscale_to_one_hot(observation)

    for row in range(trimmed_obs.shape[0]):
        for col in range(trimmed_obs.shape[1]):
            print(trimmed_obs[row, col], end="")
        print()

    # input() # Enter to advance timesteps
    print()

    if i != 0:
        print(f"Score: {additional_reward.calculate_additional_reward(last_trimmed_obs, trimmed_obs, False)}")

    if terminated or truncated:
        observation, info = env.reset()

env.close()