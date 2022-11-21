
import os
import numpy as np
import gym  # == 0.21.0
import tqdm

MAX_IMAGES = 10 ** 4
SKIP = 100
env_id = "SpaceInvaders-v4"

env = gym.make(env_id, render_mode="rgb_array")
img = env.reset()

dataset = np.zeros((MAX_IMAGES,) + img.shape)

n = 0
step = 0
with tqdm.tqdm(total=MAX_IMAGES) as pbar:
    while True and n < MAX_IMAGES:
        env.reset()
        done = False
        while not done:
            obs, r, done, _ = env.step(env.action_space.sample())
            
            if step % SKIP == 0:
                img = env.render()
                dataset[n] = img
                n += 1
                pbar.update(1)
            
            if not n < MAX_IMAGES:
                break
                
            step += 1
            

os.makedirs("dataset", exist_ok=True)
np.save(f"dataset/{env_id}", dataset)

print("finish.")
