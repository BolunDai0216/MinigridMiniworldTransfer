import gymnasium as gym
from pdb import set_trace
from stable_baselines3 import PPO
import minigrid
from minigrid.wrappers import ImgObsWrapper, ViewSizeWrapper
from gymnasium.core import ObservationWrapper
from gymnasium import spaces
import torch.nn as nn
import numpy as np
import torch


class ChannelFirstWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = env.observation_space.shape
        new_shape = (shape[2], shape[0], shape[1])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype="uint8",
        )

    def observation(self, obs):
        return obs.transpose(2, 0, 1).astype(np.float32)


def main():
    env = gym.make("MiniGrid-Empty-16x16-v0")
    env = ImgObsWrapper(env)
    env = ChannelFirstWrapper(env)
    obs, info = env.reset()

    # Conv2d(in_channel, out_channel, kernel_size)
    cnn = nn.Sequential(
        nn.Conv2d(3, 16, (2, 2)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(16, 32, (2, 2)),
        nn.ReLU(),
        nn.Conv2d(32, 64, (2, 2)),
        nn.ReLU(),
    )

    # cnn = nn.Sequential(nn.Conv2d(3, 16, (2, 2)))
    obs = torch.from_numpy(obs)

    output = cnn(obs)

    set_trace()

    # model = PPO("CnnPolicy", env, verbose=1)
    # model.learn(total_timesteps=2e5)
    # model.save("models/ppo/minigrid_empty")


if __name__ == "__main__":
    main()
