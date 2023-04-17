from pdb import set_trace

import gymnasium as gym
import minigrid
import torch
import torch.nn as nn


def main():
    # env = gym.make("MiniGrid-GoToDoor-6x6-v0")
    color_embedding = nn.Embedding(6, 8)

    color_to_idx = {
        "blue": 0,
        "green": 1,
        "grey": 2,
        "purple": 3,
        "red": 4,
        "yellow": 5,
    }

    lookup_tensor = torch.tensor([color_to_idx["purple"]], dtype=torch.long)
    purple_embedding = color_embedding(lookup_tensor)

    print(purple_embedding)


if __name__ == "__main__":
    main()
