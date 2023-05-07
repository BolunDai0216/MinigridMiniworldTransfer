import argparse
from datetime import datetime
from pdb import set_trace
from time import time

import gymnasium as gym
import minigrid
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from minigrid_gotoobj import GoToObjEnv


class ObjObsWrapper(ObservationWrapper):
    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.observation_space = Dict(
            {
                "image": env.observation_space.spaces["image"],
                "mission": Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
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

        self.obj_one_hot_dict = {
            "ball": np.array([1.0, 0.0, 0.0]),
            "box": np.array([0.0, 1.0, 0.0]),
            "key": np.array([0.0, 0.0, 1.0]),
        }

    def observation(self, obs):
        mission_array = np.concatenate(
            self.color_one_hot_dict[self.target_color],
            self.obj_one_hot_dict[self.target_obj],
        )

        wrapped_obs = {
            "image": obs["image"],
            "mission": mission_array,
        }

        return wrapped_obs


class ObjEnvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

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

                linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
                extractors["image"] = nn.Sequential(*(list(cnn) + list(linear)))
                total_concat_size += 64

            elif key == "mission":
                extractors["mission"] = nn.Linear(subspace.shape[0], 32)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument(
        "--load_model",
        default="minigrid_gotoobj_20230424-104508/iter_2000000_steps.zip",
    )
    parser.add_argument("--render", action="store_true", help="render trained models")
    args = parser.parse_args()

    policy_kwargs = dict(features_extractor_class=ObjEnvExtractor)

    # Create time stamp of experiment
    stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")

    if args.train:
        env = GoToObjEnv()
        env = ObjObsWrapper(env)

        checkpoint_callback = CheckpointCallback(
            save_freq=1e5,
            save_path=f"./models/ppo/minigrid_gotoobj_{stamp}/",
            name_prefix="iter",
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./logs/ppo/minigrid_gotoobj_tensorboard/",
        )
        model.learn(
            2e6,
            tb_log_name=f"{stamp}",
            callback=checkpoint_callback,
        )
    else:
        if args.render:
            env = GoToObjEnv(render_mode="human")
        else:
            env = GoToObjEnv()
        env = ObjObsWrapper(env)

        ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

        # add the experiment time stamp
        ppo = ppo.load(f"models/ppo/{args.load_model}", env=env)

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
