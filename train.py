# Step 2: Train the model

# Callback

import os # for file path management

from web_game_env import WebGame # custom environment

from stable_baselines3.common.callbacks import BaseCallback # for saving the model
from stable_baselines3.common import env_checker # to check if the environment is valid

# Periodically saves the model during training to ensure progress is not lost
class TrainAndLoggingCallback(BaseCallback):
    # define how often to save and where to save the files
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    # runs once before training starts and creates the training folder
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    # runs every training step
    def _on_step(self):
        # check whether the current step count has reached a multiple of check_freq (means it is time to save the model)
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, "best_model_{}".format(self.n_calls)) # e.g. best_model_1000
            self.model.save(model_path)
        # tell stable baselines to keep training
        return True

CHECKPOINT_DIR = "./DinoAI/train/"
LOGS_DIR = "./DinoAI/logs/"

# Save the model every 1000 steps
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

# Create environment
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # for frame stacking

# # raw_env = WebGame()
# # print(env_checker.check_env(raw_env)) # check if the environment is valid
# # raw_env.close()

# # Wrap the environment in a DummyVecEnv, which is required for stable baselines to do frame stacking
# env = DummyVecEnv([lambda: WebGame()])
# env = VecFrameStack(env, n_stack=4) # stack 4 frames together to give the model a sense of motion and speed

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

env = DummyVecEnv([lambda: Monitor(WebGame())])
env = VecFrameStack(env, n_stack=4)

# Build DQN and train

from stable_baselines3 import DQN # deep-Q network algorithm

# Create model
model = DQN(
    policy="CnnPolicy",
    env=env, # gym custom web env
    tensorboard_log=LOGS_DIR,
    device="cuda", # use GPU for training
    verbose=1, # logging results
    buffer_size= 50_000, # how many frames we collect inside the DQN buffer
    learning_starts=1_000, # start learning after 1000 steps
    learning_rate=1e-4, # start learning after 1000 steps
    batch_size=32, # how many samples to learn from at each step
    train_freq=4, # how often to train the model (every 4 steps)
    target_update_interval=1_000, # how often to update the target network
    exploration_fraction=0.4, # what fraction of the training period to spend on exploration (vs exploitation)
    exploration_initial_eps=1.0, # initial exploration rate
    exploration_final_eps=0.05, # final exploration rate
)

# model = DQN.load("./DinoAI/train/dino_ai_final", env=env, device="cuda") # load the last model from training so far to continue training from there

# Training
model.learn(
    total_timesteps=20_000, # how long to train for
    callback=callback,
    reset_num_timesteps=False, # continue training from the previous model (do not reset the timestep count)
    tb_log_name="Optimized_Test_2_Logs" # name of the tensorboard log file
)

model.save("./DinoAI/train/dino_ai_final")