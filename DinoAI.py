from mss import mss # used for screen capture
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
    # set up environment, actions, and observation shapes
    def __init__(self):
        # subclass model
        super().__init__()

        # setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8) # img which is 83x100 pixels
        self.action_space = Discrete(3) # 3 actions: jump, duck, do nothing

    # what gets called to do something in the game
    def step(self, action):
        # Action key
        # 0: space(up), 1: duck(down), 2: do nothing(No op)


        pass
    # render function: visualize the game
    def render(self):
        pass
    # restart game
    def reset(self):
        pass
    # closes down the observation
    def close(self):
        pass
    # get a specific part of the observation of the game that we want
    def get_observation(self):
        pass
    # checks if the game is over
    def get_done(self):
        pass

env = WebGame()

print(env.action_space.sample()) # test action space
print(env.observation_space.sample()) # test observation space
plt.imshow(env.observation_space.sample()[0]) # test observation capture
plt.show()


# Test environment