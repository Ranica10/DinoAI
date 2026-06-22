from mss import MSS # used for screen capture
import cv2 # used for image processing

import numpy as np # transformational framework
import time # for pausing
from matplotlib import pyplot as plt # visualize captured frames

import pytesseract # OCR for game over detection
import pydirectinput # used for sending key presses to the game 

# Environment components
from gymnasium import Env # base class for our custom environment
from gymnasium.spaces import Discrete, Box # box: rep shape of the state space, discrete: rep possible actions in the game

# Step 1: Build the environment

# Create environment
class WebGame(Env):
    # Set up environment, actions, and observation shapes
    def __init__(self):
        # subclass model
        super().__init__()

        # setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8) # img which is 83x100 pixels
        self.action_space = Discrete(3) # 3 actions: jump, duck, do nothing

        # define extraction parameters for screen capture
        self.cap = MSS()
        self.game_region = {'top': 300, 'left': 0, 'width': 600, 'height': 700} # region of the screen where the game is located
        self.game_over_region = {'top': 405, 'left': 630, 'width': 660, 'height': 70} # region where "Game Over" text appears

    # Step function: take an action and return the next observation, reward, done, and info
    def step(self, action):
        # Action key
        # 0: space(up), 1: duck(down), 2: do nothing(No op)
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }

        # If no operation is given, do the action
        if action != 2:
            # press the key specified for the action via pydirectinput
            pydirectinput.press(action_map[action])

        # Check whether the game is done
        done, done_cap = self.get_done()
        # Get the next observation
        next_observation = self.get_observation()

        # For every frame the dino is alive, reward the agent +1
        reward = 1

        terminated = done
        truncated = False

        # Info dictionary (for stable baselines)
        info = {}

        return next_observation, reward, terminated, truncated, info

    # Render function: visualize the game
    def render(self):
        # show the game frame (which is the current frame)
        cv2.imshow('Game', np.array(self.cap.grab(self.game_region))[:,:,:3].astype(np.uint8))
        
        # If the "q" key is pressed, call the close function
        if cv2.waitKey(0) & 0xFF == ord("q"):
            self.close()

    # Closes down the observation
    def close(self):
        cv2.destroyAllWindows()

    # Restart game
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        time.sleep(1) # wait 1 second
        
        pydirectinput.click(x=150,y=150) # click at any point on the screen
        pydirectinput.press("space") # press space to restart the game

        obs = self.get_observation()
        info = {}

        # return the next observation of the game after restarting
        return obs, info
    
    # Get a specific part of the observation of the game that we want
    def get_observation(self):
        # get screen capture of the game region and convert to numpy array
        raw = np.array(self.cap.grab(self.game_region))[:,:,:3].astype(np.uint8) # keep only the RGB channels and convert to uint8

        # grayscale 
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resize to 83x100
        resized = cv2.resize(gray, (100, 83))
        # reshape 
        channel = np.reshape(resized, (1, 83, 100))

        # return the processed observation
        return channel

    # Checks if the game is over
    def get_done(self):
        # get done screen
        done_cap = np.array(self.cap.grab(self.game_over_region))[:,:,:3].astype(np.uint8) # keep only the RGB channels and convert to uint8

        # define valid game over text
        done_strings = ["GAME", "GAHE"]

        # apply OCR to detect game over text
        done = False
        result = pytesseract.image_to_string(done_cap)[:4] # get the first 4 characters of extracted text

        if result in done_strings:
            done = True

        return done, done_cap

# Test environment

env = WebGame()

# Play 10 games
for episode in range(1):
    # Start off with a new game
    obs, info = env.reset()
    done = False
    
    # Counter for total rewards
    total_rewards = 0

    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        
        total_rewards += reward # increment reward
    
    print(f"Total reward for episode {episode}: {total_rewards} \n")

# obs = env.get_observation()
# plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_GRAY2RGB)) # test observation capture
# plt.show()

# result, done, done_cap =  env.get_done()

# print(done)
# print(result)
# plt.imshow(done_cap)
# plt.show()

# env.reset()


# Step 2: Train the model

# Callback

import os # for file path management

from stable_baselines3.common.callbacks import BaseCallback # for saving the model
from stable_baselines3.common import env_checker # to check if the environment is valid

print(env_checker.check_env(env)) # check if the environment is valid


# Build DQN and train
