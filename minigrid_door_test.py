from pdb import set_trace

import gymnasium as gym
import minigrid
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box, Dict, MultiBinary
from minigrid.manual_control import ManualControl
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from minigrid_cnn import MinigridCNN

TRAIN = True


class DoorObsWrapper(ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.observation_space = Dict(
            {
                "image": env.observation_space.spaces["image"],
                "door_color": Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

        self.color_one_hot_dict = {
            "red": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "green": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "blue": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "purple": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "yellow": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "grey": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        }

    def observation(self, obs):
        wrapped_obs = {
            "image": obs["image"],
            "door_color": self.color_one_hot_dict[self.target_color],
        }

        return wrapped_obs


class DoorEnvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict, features_dim: int = 144):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim)

        extractors = {}
        total_concat_size = 0

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                cnn = nn.Sequential(
                    nn.Conv2d(3, 16, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, (2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]

                linear = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())
                extractors["image"] = nn.Sequential(*(list(cnn) + list(linear)))
                total_concat_size += 128

            elif key == "door_color":
                extractors["door_color"] = nn.Linear(subspace.shape[0], 32)
                total_concat_size += 32

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def main():
    policy_kwargs = dict(
        features_extractor_class=DoorEnvExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    if TRAIN:
        env = gym.make("MiniGrid-GoToDoor-8x8-v0")
        env = DoorObsWrapper(env)
        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        model.learn(2e5)
        model.save("models/ppo/minigrid_door")
    else:
        env = gym.make("MiniGrid-GoToDoor-5x5-v0", render_mode="human")
        env = DoorObsWrapper(env)

    ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    ppo = ppo.load("models/ppo/minigrid_door", env=env)

    obs, info = env.reset()
    rewards = 0

    for i in range(2000):
        action, _state = ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward

        if terminated or truncated:
            print(f"Test reward: {rewards}")
            obs, info = env.reset()
            rewards = 0
            continue

    print(f"Test reward: {rewards}")

    env.close()


if __name__ == "__main__":
    main()
