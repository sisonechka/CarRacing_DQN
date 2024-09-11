import numpy as np

# Prepares and cleans images for input into neural networks.
def process_image(image):
    # Adjust the red and green channels of the image.
    red = image[:,:,0:1] * 0.55  # Dims the red tones.
    green = image[:,:,1:2] * -0.45 + 255 * 0.495  # Adjusts the green tones to be brighter.
    
    # Combine the adjusted red and green channels.
    image = np.squeeze(red + green) / 255  # Normalize and combine the channels.
    
    # Remove parts of the image that are too dark (unimportant details).
    image[image < 0.4] = 0
    
    # Return the processed image.
    return image

class modified_env:
    def __init__(self, env, render=False, timestep=5, init_bad_steps=0, bad_step_limit=17, early_stop=True):
        self.env = env
        self.env.reset()
        self.timestep = timestep
        self.render = render
        self.bad_steps = init_bad_steps
        self.bad_step_limit = bad_step_limit
        self.ep_len = 0
        self.real_rew = 0
        self.early_stop = early_stop
        
        if render:
            self.env.render()
   
    # Resets the environment and prepares it for a new episode.
    def reset(self):
        self.ep_len = 0
        self.real_rew = 0
        state = self.env.reset()
        return state
  
    # Skips unnecessary episodes, focusing on the important parts of training.
    def skip_episodes(self, num_episodes, action_code):
        for i in range(num_episodes):
            state, rew, done, info = self.env.step(action_code)[:4]  # Take action and get state, reward, done, info
            self.real_rew += rew  # Add the reward to the total reward for the episode.
            self.ep_len += 1  # Increment the episode length counter.
            if done:  # Check if the episode is done.
                break
        return process_image(state), rew, done, info  # Return the final state, reward, done flag, and info.

    # Takes an action in the environment, updating the state and accumulating rewards.
    def step(self, action_code):
        step_rew = 0  # Initialize the reward for this step.

        # Perform the action multiple times to simulate continuous behavior.
        for i in range(self.timestep):
            state, rew, done, info = self.env.step(action_code)[:4]  # Take action and get state, reward, done, info
            self.real_rew += rew  # Add the reward to the total reward for the episode.
            self.ep_len += 1  # Increment the episode length counter.
            
            step_rew += rew  # Accumulate the reward for this step.

            # Track bad steps (negative rewards).
            if rew < 0:
                self.bad_steps += 1
            else:
                self.bad_steps = 0  # Reset bad steps counter if a positive reward is received.

            # If too many bad steps occur, and early stopping is enabled, end the episode.
            if self.bad_steps >= self.bad_step_limit and self.early_stop:
                self.bad_steps = 0  # Reset the bad steps counter.
                done = True  # End the episode.
                step_rew += -100  # Apply a penalty for early termination.

            if done:  # If the episode is done, stop taking further steps.
                break

        return process_image(state), step_rew, done, info  # Return the processed state, total step reward, done flag, and info.
