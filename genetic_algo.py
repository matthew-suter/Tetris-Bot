import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Rendering
from IPython import display as ipythondisplay
from PIL import Image

# Additional reward code
import additional_reward

import time

env = gym.make("ALE/Tetris-v5", obs_type="grayscale")
# render_env = gym.make("ALE/Tetris-v5", obs_type="grayscale", render_mode="human")
observation, info = env.reset()

# Hyperparameters
num_hidden_units = 400
mutation_factor = 0.01

# TODO: Implement these!
bonus_mutation_factor = 0.1
bonus_mutation_score_max = 1500

num_actors = 10
best_keep = 2 # Keep the best N actors to the next generation, kill the rest

num_generations = 5

image_frame_decimation = 5 # The steps between frames when saving as an image

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
    # result = tf.reshape(result, [1, -1])
    return result


def training():
    assert num_actors % best_keep == 0, "An even proportion of the population must be saved!"
    actors = []

    for i in range(num_actors):
        actors.append(TetrisActor(num_actions=env.action_space.n, num_hidden_units=num_hidden_units))

    for generation_num in range(num_generations):
        print(f"\nGeneration {generation_num+1}")

        # Test all of the actors
        filepath = f"genetic_previews/gen_{generation_num+1}.gif"
        actor_scores = trial_actors(actors, verbose_printing=True, render_game=True, render_filename=filepath)
        print(np.sort(actor_scores))

        # Take the best N actors
        new_actors = []
        best_actors_idx = np.argpartition(actor_scores, -best_keep)[-best_keep:]
        # print(f"Best actor indices: {best_actors_idx}")
        for idx in best_actors_idx:
            new_actors.append(actors[idx])
        
        for i in range(num_actors-best_keep):
            new_actors.append(tf.keras.models.clone_model(new_actors[i%best_keep]))

        # Mutate the duplicated actors
        actors = new_actors
        for actor_idx in range(best_keep, len(actors)):
            actors[actor_idx].shuffle_weights(mutation_factor)



def trial_actors(actors, verbose_printing=False, render_game=False, render_filename="Tetris_Game"):
    """
    Trials all actors in a supplied list

    Arguments:
        actors: list of TetrisActor layers
        verbose_printing: print verbosely
        render_game: Save a .gif of the first actor's playthrough
        render_filename: The path the gif is saved to
    """
    actor_scores = np.zeros(len(actors))

    if render_game and not render_filename.endswith(".gif"):
        render_filename += ".gif"
    

    # CPU time tracking
    track_timing = True

    total_time = 0
    game_time  = 0
    model_time = 0

    total_time_start = time.process_time_ns()
    game_time_start = 0
    model_time_start = 0


    for i, actor in enumerate(actors):
        # Game start
        if verbose_printing:
            print(f"Training actor #{i+1:02}... ", end="")
        greyscale, info = env.reset()
        state = greyscale_to_one_hot(greyscale)

        # Game control and scoring
        done = False
        truncated = False
        steps_survived = 0
        cumulative_score = 0

        # Rendering init
        if render_game and i==0:
            images = [Image.fromarray(greyscale)]

        while not (done or truncated):
            last_state = state
            state = greyscale_to_one_hot(greyscale)

            if track_timing:
                model_time_start = time.process_time_ns()
            action = actor.call(tf.reshape(state, [1, -1]))
            if track_timing:
                model_time += time.process_time_ns() - model_time_start

            # print(actor.summary())
            action = int(tf.argmax(action, axis=1))

            if track_timing:
                game_time_start = time.process_time_ns()
            greyscale, reward, done, truncated, info = env.step(action)
            if track_timing:
                game_time += time.process_time_ns() - game_time_start

            steps_survived += 1
            cumulative_score += additional_reward.calculate_additional_reward(last_state, state, False)
        
            # Render screen every 10 steps
            if render_game and i==0 and (steps_survived % image_frame_decimation == 0):
                images.append(Image.fromarray(greyscale))
        
        if render_game:
            # loop=0: loop forever, duration=1: play each frame for 1ms
            images[0].save(render_filename, save_all=True, append_images=images[1:], loop=0, duration=1)


        if verbose_printing:
            print(f"Lasted {steps_survived} steps, scored {cumulative_score} points")
        
        actor_scores[i] = cumulative_score
    

    if track_timing:
        total_time = time.process_time_ns() - total_time_start

        print(f"\n\nTotal time: {total_time/1e9:0.2f} sec \nModel time: {model_time/1e9:0.2f} sec ({100*model_time/total_time:0.2f}%) \nGame time:  {game_time/1e9:0.2f} sec ({100*game_time/total_time:0.2f}%) \n")


    return actor_scores



if __name__ == "__main__":
    for i in range(50):
        print()
    training()
