import tensorflow as tf
import numpy as np

def calculate_additional_reward(previous_grid, current_grid, done):
    # there are 5 ways of getting a reward:
    # 1. not dying
    # 2. aggregate heigh
    # 3. complete lines
    # 4. holes
    # 5. bumpiness

    #Coeffients for each reward
    #got this off the interwebs, might need to be adjusted
    a,b,c,d,e = 0.1,0.51,0.76,0.36,0.18

    reward = 0
    heighty_reward = 0
    holey_reward = 0
    Bumpi_reward = 0
    liney_reward = 0

    #not dying
    if not done:
        # Reward is 1 for each step the game continues
        dony_reward = 0.01
    else:
        # Significant penalty for losing the game.
        dony_reward = -100
    # Implement the actual logic based on the game state
    # Add rewards or penalties based on game state, e.g., line clearance, height, etc.

    # Aggregate height
    current_height = np.max(tf.reduce_sum(current_grid, axis=0))
    previous_height = np.max(tf.reduce_sum(previous_grid, axis=0))
    if current_height > previous_height:
        heighty_reward -= 1
    
    # Complete lines
    if tf.reduce_sum(previous_grid) > tf.reduce_sum(current_grid):
        liney_reward += 10

    # Holes
    #copilot code, likely gonna break, or eat be O(n^5000)
    # for i in range(current_grid.shape[1]):
    #     for j in range(current_grid.shape[0]):
    #         if current_grid[j,i] == 1 and previous_grid[j,i] == 0:
    #             for k in range(j+1, current_grid.shape[0]):
    #                 if current_grid[k,i] == 0:
    #                     holey_reward -= 1

    # Bumpiness
    # for i in range(current_grid.shape[1]-1):
    #     Bumpi_reward -= abs(sum(current_grid[:,i]) - sum(current_grid[:,i+1]))

    return a*dony_reward + b*heighty_reward #+ c*liney_reward + e*Bumpi_reward
