import gymnasium as gym
from pdb import set_trace
from stable_baselines3 import PPO
import miniworld
from gymnasium.core import ObservationWrapper
from gymnasium import spaces


def main():
    env = gym.make("MiniWorld-OneRoom-v0")
    obs, info = env.reset()

    set_trace()

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=2e5)
    model.save("models/ppo/miniworld_oneroom")


if __name__ == "__main__":
    main()
