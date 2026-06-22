import os
import time

from stable_baselines3 import DQN # deep-Q network algorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from web_game_env import WebGame # custom 

# Step 3: Test the model

# Load final model from training

MODEL_PATH = "./DinoAI/train/dino_ai_final" # path to the model we want to load

# Create environment and load the model
env = DummyVecEnv([lambda: WebGame()])
env = VecFrameStack(env, n_stack=4)

model = DQN.load(MODEL_PATH, env=env) # load the model and pass in the environment so it can be used for prediction

# Play 10 games
for episode in range(10):
    # Start off with a new game
    obs = env.reset()
    done = [False]
    
    # Counter for total rewards
    total_rewards = 0

    while not done:
        # get the action from the model and take a step in the environment
        action, _ = model.predict(obs, deterministic=True) # deterministic: use the action with the highest predicted reward
        obs, reward, done, info = env.step(action)

        total_rewards += reward # increment reward
        time.sleep(0.01)
    
    print(f"Total reward for episode {episode}: {total_rewards} \n")
    time.sleep(2)