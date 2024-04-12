import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

env = gym.make("ALE/Tetris-v5", obs_type="grayscale")
observation, info = env.reset()

# Hyperparameters
num_hidden_units = 50
mutation_factor = 0.01
num_actors = 10
best_keep = 5 # Keep the best N actors to the next generation, kill the rest

num_generations = 5

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


    def shuffle_weights(self, shuffle_factor=0.01):
        """Shuffles the values of all layers' weights"""
        dense1_weights = self.dense1.get_weights()
        self._shuffle_layer_weights(dense1_weights, shuffle_factor)
        self.dense1.set_weights(dense1_weights)

        dense2_weights = self.dense2.get_weights()
        self._shuffle_layer_weights(dense2_weights, shuffle_factor)
        self.dense2.set_weights(dense2_weights)


    def _shuffle_layer_weights(self, layer_weights, shuffle_factor):
        """Layer weights are a python list of numpy arrays. Gross, this scales each value by a random factor"""
        for i in range(len(layer_weights)):
            scale_factor = tf.random.uniform(layer_weights[i].shape, minval=1-shuffle_factor, maxval=1+shuffle_factor)
            scaled = layer_weights[i] * scale_factor
            layer_weights[i] = scaled


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
    result = tf.reshape(result, [1, -1])
    return result



def training():
    actors = []

    for i in range(num_actors):
        actors.append(TetrisActor(num_actions=env.action_space.n, num_hidden_units=num_hidden_units))

    for i in range(num_generations):
        print(f"Generation {i+1}")

        # Test all of the actors
        actor_scores = trial_actors(actors, verbose_printing=True)
        print(actor_scores)

        # Take the best N actors
        new_actors = []
        best_actors_idx = np.argpartition(actor_scores, -best_keep)[-best_keep:]
        print(best_actors_idx)

        for actor in actors:
            actor.shuffle_weights(mutation_factor)



def trial_actors(actors, verbose_printing=False):
    actor_scores = np.zeros(len(actors))

    for i, actor in enumerate(actors):
        if verbose_printing:
            print(f"Training actor #{i+1}... ", end="")
        greyscale, info = env.reset()

        done = False
        truncated = False
        steps_survived = 0

        while not (done or truncated):
            state = greyscale_to_one_hot(greyscale)
            action = actor.call(state)

            # print(actor.summary())
            action = int(tf.argmax(action, axis=1))
            greyscale, reward, done, truncated, info = env.step(action)
            steps_survived += 1
        
        if verbose_printing:
            print(f"Lasted {steps_survived} steps")
        actor_scores[i] = steps_survived
    
    return actor_scores




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