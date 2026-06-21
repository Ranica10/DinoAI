from mss import mss # used for screen capture
import cv2 # used for image processing

import numpy as np # transformational framework
import time # for pausing
from matplotlib import pyplot as plt # visualize captured frames

import pytesseract # OCR for game over detection
import pydirectinput # used for sending key presses to the game 

# Environment components
from gym import Env # base class for our custom environment
from gym.spaces import Discrete, Box # box: rep shape of the state space, discrete: rep possible actions in the game

# Step 1: Build the environment

# Create environment


# Test environment