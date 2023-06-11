from pdb import set_trace

import gymnasium as gym
import miniworld
from stable_baselines3 import PPO


def main():
    """
    Create a virtual display using

    xvfb-run python3 miniworld_test.py
    """
    env = gym.make("MiniWorld-OneRoom-v0")

    model = PPO("CnnPolicy", env, verbose=1)
    model = model.load("models/ppo/miniworld_oneroom", env=env)
    model.learn(total_timesteps=1e4)
    model.save("models/ppo/miniworld_oneroom")


if __name__ == "__main__":
    main()
