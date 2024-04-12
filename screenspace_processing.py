import gymnasium as gym

env = gym.make("ALE/Tetris-v5", render_mode="human", obs_type="grayscale")
observation, info = env.reset()


def greyscale_to_one_hot(greyscale):
    grid =  greyscale[28:204:8, 24:64:4] # Trim to only include the playspace, shifting one pixel down and left to avoid inter-block gaps, and then downscale
    grid = (grid != 0x6F)*1 # Check if background colour (Hex code 0x6F), and convert to numerical boolean using disgusting typecasting
    return grid

def calculate_additional_reward(previous_grid, current_grid, done):
    # there are 5 ways of getting a reward:
    # 1. not dying
    # 2. aggregate heigh
    # 3. complete lines
    # 4. holes
    # 5. bumpiness

    #Coeffients for each reward
    #got this off the interwebs, might need to be adjusted
    a,b,c,d,e = 0.1,-0.51,0.76,-0.36,-0.18

    reward = 0

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
    current_height = max([sum(current_grid[:,i]) for i in range(current_grid.shape[1])])
    previous_height = max([sum(previous_grid[:,i]) for i in range(previous_grid.shape[1])])
    if current_height > previous_height:
        heighty_reward -= 1
    
    # Complete lines
    if sum(previous_grid) > sum(current_grid):
        liney_reward += 10

    # Holes
    #copilot code, likely gonna break, or eat be O(n^5000)
    for i in range(current_grid.shape[1]):
        for j in range(current_grid.shape[0]):
            if current_grid[j,i] == 1 and previous_grid[j,i] == 0:
                for k in range(j+1, current_grid.shape[0]):
                    if current_grid[k,i] == 0:
                        holey_reward -= 1

    # Bumpiness
    for i in range(current_grid.shape[1]-1):
        Bumpi_reward -= abs(sum(current_grid[:,i]) - sum(current_grid[:,i+1]))

    return a*dony_reward + b*heighty_reward + c*liney_reward + d*holey_reward + e*Bumpi_reward


# def scored_this_turn()


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