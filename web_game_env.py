from mss import MSS # used for screen capture
import cv2 # used for image processing

import numpy as np # transformational framework
import time # for pausing
from matplotlib import pyplot as plt # visualize captured frames

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
        self.game_region = {'top': 350, 'left': 0, 'width': 600, 'height': 450} # region of the screen where the game is located
        self.game_over_region = {'top': 405, 'left': 630, 'width': 330, 'height': 70} # region where "Game Over" text appears

        self.step_count = 0
        self.last_done = False

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
        
        self.step_count += 1

        # To optimize performance, check for the "done" state every 5 steps
        if self.step_count % 5 == 0:
            # Check whether the game is done
            done, done_cap = self.get_done()
            self.last_done = done
        else:
            done = self.last_done

        # Get the next observation
        next_observation = self.get_observation()

        # For every frame the dino is alive, reward the agent +1

        # Penalize unnecessary ducking (action 1) when not needed
        if action == 1:
            reward = 0.5  # ducking is less "good" than doing nothing or jumping
        elif not done:
            reward = 1 + (self.step_count * 0.001)  # increasing reward over time
        else:
            reward = -10 # if done, give a negative reward to encourage the agent to avoid dying

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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()

    # Closes down the observation
    def close(self):
        cv2.destroyAllWindows()

    # Restart game
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0 # reset step count
        self.last_done = False # reset done state

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

        # apply thresholding to convert the image to black/white
        _, threshold = cv2.threshold(
            gray,
            180,
            255,
            cv2.THRESH_BINARY_INV
        )

        # resize to 83x100
        resized = cv2.resize(threshold, (100, 83))
        # reshape 
        channel = np.reshape(resized, (1, 83, 100))

        # return the processed observation
        return channel

    # Checks if the game is over
    def get_done(self):
        # grab the game over region and convert to grayscale
        done_cap = np.array(self.cap.grab(self.game_over_region))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(done_cap, cv2.COLOR_BGR2GRAY)

        # crop only the exact area where GAME OVER appears inside done_cap (to avoid birds and other objects in the background)
        roi = gray[20:70, 30:330]

        # detect dark gray GAME OVER letters
        mask = roi < 140

        # create a mask where dark pixels are True
        ys, xs = np.where(mask)

        # if there are no dark pixels at all, then GAME OVER is definitely not shown
        if len(xs) == 0:
            return False, done_cap

        # calculate the width and height of the detected dark object/text
        text_width = xs.max() - xs.min() # the diff btwn leftmost and rightmost pixels gives the width
        # text_height = ys.max() - ys.min() # height

        roi_h, roi_w = roi.shape # get the height and width of the ROI
        
        width_ratio = text_width / roi_w # calc how much of the cropped region width is covered by dark pixels
        # height_ratio = text_height / roi_h # height
        
        # calculate what percentage of the crop is dark pixels
        dark_ratio = np.sum(mask) / mask.size

        # print("dark:", dark_ratio, "width:", width_ratio, "height:", height_ratio)

        # the game is only done if all 2 conditions are met
        done = (
            dark_ratio > 0.15 and # at least 15% of the cropped region is dark pixels
            width_ratio > 0.75 # at least 75% of the cropped region width is covered by dark pixels
        )

        return done, done_cap

# Test environment

# env = WebGame()

# #obs = env.render()

# obs = env.get_observation()
# plt.imshow(obs[0])
# plt.show()

# plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_GRAY2RGB)) # test observation capture
# plt.show()

# env = WebGame()

# done, done_cap =  env.get_done()

# print(done)

# plt.imshow(done_cap)
# plt.show()

# env.reset()