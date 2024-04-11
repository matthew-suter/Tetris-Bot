import gymnasium as gym
from tetris_agent import *

env = gym.make("ALE/Tetris-v5", render_mode="human", obs_type="grayscale")
observation, info = env.reset()

player = TetrisAgent()
print(player)

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    # print(observation.shape)
    # print()
    # print(observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()