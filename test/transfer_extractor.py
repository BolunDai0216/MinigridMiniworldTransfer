import torch as th
import torch.nn as nn
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GridEnvExtractor(BaseFeaturesExtractor):
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
                    nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
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

            elif key == "door_color":
                extractors["door_color"] = nn.Linear(subspace.shape[0], 32)
                extractors["door_color"].weight.requires_grad = False
                extractors["door_color"].bias.requires_grad = False
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
