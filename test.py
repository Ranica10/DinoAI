import os
import time

from stable_baselines3 import DQN # deep-Q network algorithm
from web_game_env import WebGame # custom 

# Step 3: Test the model

# Load final model from training

MODEL_PATH = "./DinoAI/train/best_model_8000" # path to the model we want to load

env = WebGame() # create environment instance to pass into the model for prediction

model = DQN.load(MODEL_PATH, env=env) # load the model and pass in the environment so it can be used for prediction

# Play 10 games
for episode in range(10):
    # Start off with a new game
    obs, info = env.reset()
    done = False
    
    # Counter for total rewards
    total_rewards = 0

    while not done:
        # get the action from the model and take a step in the environment
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(int(action))
        time.sleep(0.01)

        done = terminated or truncated # check if done
        total_rewards += reward # increment reward
    
    print(f"Total reward for episode {episode}: {total_rewards} \n")
    time.sleep(2)