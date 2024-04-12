import collections
import gymnasium as gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

#Hyperparmeters
min_episodes_criterion = 1000
max_episodes = 10 #10000
max_steps_per_episode = 20000
learning_rate = 0.01

# Epsilon-Greedy Hyperparameters
#These adjust the random behavior of the AI, allowing it to explore more
epsilon = 1.0  # Initial epsilon value
epsilon_min = 0.01  # Minimum value of epsilon
epsilon_decay = 0.995  # Decay rate for epsilon

# Create the environment
env = gym.make("ALE/Tetris-v5", obs_type="ram")

# Set seed for experiment reproducibility
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

# %%

@tf.keras.utils.register_keras_serializable()
class TetrisActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self,
      num_actions: int,
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)
  
  def get_config(self):
    # Get the base configuration from the parent class
    config = super().get_config()
    # Add custom parameters to the config
    config['num_actions'] = self.actor.units
    config['num_hidden_units'] = self.common.units
    return config

  # def get_config(self):
  #   base_config = super().get_config()
  #   config = {}
  #   # config = {
  #   #     "sublayer": tf.keras.saving.serialize_keras_object(self.sublayer),
  #   # }
  #   return {**base_config, **config}


# %%

# Wrap Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def calculate_additional_reward(state, done):

    if not done:
        # Reward is 1 for each step the game continues
        reward = 1
    else:
        # Significant penalty for losing the game.
        reward = -100
    # Implement the actual logic based on the game state
    # Add rewards or penalties based on game state, e.g., line clearance, height, etc.
    return reward

@tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32])
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, truncated, info = env.step(action)

  additional_reward = calculate_additional_reward(state, done)
  reward += additional_reward

  return (state.astype(np.float32),
          np.array(reward, np.int32),
          np.array(done, np.int32))


# %%
def run_episode(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    #action = env.action_space.sample()
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = env_step(action)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  print(action_probs, values, rewards)

  return action_probs, values, rewards


# %%
def get_expected_return(
    rewards: tf.Tensor,
    gamma: float,
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) /
               (tf.math.reduce_std(returns) + eps))

  return returns


# %%
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined Actor-Critic loss."""

  #entropy should encorage the AI to act randomly, allowing it to explore more.
  entropy = tf.reduce_sum(-action_probs * tf.math.log(action_probs + 1e-9), axis=1)


  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss - 0.01 * entropy


# %%
optimizer = tf.keras.optimizers.Adam(learning_rate)


@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode)

    # Calculate the expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

    # Calculate the loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward

# %%time

def choose_action(state, model, epsilon):
  #This selects what action to take, either randomly or based on the model's prediction
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Choose a random action
    else:
        q_values = model(state)  # Assuming model(state) directly gives Q-values
        return np.argmax(q_values)  # Choose the best action based on the model's prediction


def train_model(model: TetrisActorCritic, save_filename):

  running_reward = 0

  # The discount factor for future rewards
  gamma = 0.99

  # Keep the last episodes reward
  episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

  t = tqdm.trange(max_episodes)
  for i in t:
      initial_state, info = env.reset()
      initial_state = tf.constant(initial_state, dtype=tf.float32)

      #action = choose_action(initial_state, model, epsilon)
      action = env.action_space.sample()
      next_state, reward, done, truncated, info = env.step(action)
      next_state = tf.constant(next_state, dtype=tf.float32)


      episode_reward = int(train_step(
          initial_state, model, optimizer, gamma, max_steps_per_episode))

      episodes_reward.append(episode_reward)
      initial_state = next_state
      running_reward = statistics.mean(episodes_reward)

      # Update the progress bar in the command line
      t.set_postfix(
          episode_reward=episode_reward, running_reward=running_reward)

      # Show the average episode reward every 10 episodes
      if i % 10 == 0:
        pass # print(f'Episode {i}: average reward: {avg_reward}')

  print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

  model.save(save_filename)
  print(f"Saved model to {save_filename}\n")

save_filename = "tetris_model.keras"

load_last_model = input("Use last model? (Y/n) ").lower() != "n"
if load_last_model:
  print("Loading model...")
  model = tf.keras.models.load_model(save_filename)
else:
  num_actions = env.action_space.n  # 2
  num_hidden_units = 128
  model = TetrisActorCritic(num_actions, num_hidden_units)
  train_model(model, save_filename)


# %%
# Render an episode and save as a GIF file

from IPython import display as ipythondisplay
from PIL import Image


play_env = gym.make("ALE/Tetris-v5", render_mode = "rgb_array", obs_type="ram")
# play_env = gym.make("ALE/Tetris-v5", obs_type="ram", repeat_action_probability=0, frameskip=999999)
# render_env = gym.make("ALE/Tetris-v5", render_mode='rgb_array', repeat_action_probability=0, frameskip=999999)

def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int):
  state, info = env.reset()
  env.reset()
  state = tf.constant(state, dtype=tf.float32)
  screen = env.render()
  images = [Image.fromarray(screen)]

  for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, 0)
    action_probs, _ = model(state)
    action = np.argmax(np.squeeze(action_probs))

    state, reward, done, truncated, info = play_env.step(action)
    state = tf.constant(state, dtype=tf.float32)

    # Render screen every 10 steps
    if i % 10 == 0:
      screen = env.render()
      images.append(Image.fromarray(screen))

    if done:
      break

  return images

# Save GIF image
images = render_episode(play_env, model, max_steps_per_episode)
image_file = 'tetris-v2.gif'
# loop=0: loop forever, duration=1: play each frame for 1ms
images[0].save(
    image_file, save_all=True, append_images=images[1:], loop=0, duration=1)
