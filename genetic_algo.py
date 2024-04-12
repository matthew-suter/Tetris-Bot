import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

env = gym.make("ALE/Tetris-v5", obs_type="grayscale")
observation, info = env.reset()

# Hyperparameters
num_hidden_units = 50

@tf.keras.utils.register_keras_serializable()
class TetrisActor(tf.keras.Model):
  """GA_Suitable Actor"""

  def __init__(
      self,
      num_actions: int,
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.dense1 = layers.Dense(num_hidden_units, activation="relu")
    self.dense2 = layers.Dense(num_actions)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    x = self.dense1(inputs)
    return self.dense2(x)
  
  def get_config(self):
    # Get the base configuration from the parent class
    config = super().get_config()
    # Add custom parameters to the config
    config['num_actions'] = self.dense2.units
    config['num_hidden_units'] = self.dense1.units
    return config


def greyscale_to_one_hot(greyscale):
    result =  greyscale[28:204:8, 24:64:4] # Trim to only include the playspace, shifting one pixel down and left to avoid inter-block gaps, and then downscale
    result = (result != 0x6F)*1 # Check if background colour (Hex code 0x6F), and convert to numerical boolean using disgusting typecasting
    return result



def training():
    num_actors = 100
    actors = []

    for i in range(num_actors):
        actors.append(TetrisActor(num_actions=env.action_space.n, num_hidden_units=num_hidden_units))

    trial_actors(actors)



def trial_actors(actors):
    for i, actor in enumerate(actors):
        print(f"Training actor #{i}...")
        greyscale, info = env.reset()

        done = False
        truncated = False

        while not (done or truncated):
            state = greyscale_to_one_hot(greyscale)
            action = actor.call(state)
            print(action)
            action = tf.argmax(action)
            print(action)
            greyscale, reward, done, truncated, info = env.step(action)




training()









# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     trimmed_obs = greyscale_to_one_hot(observation)

#     for row in range(trimmed_obs.shape[0]):
#         for col in range(trimmed_obs.shape[1]):
#             print(trimmed_obs[row, col], end="")
#         print()

#     # input() # Enter to advance timesteps
#     print()

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()