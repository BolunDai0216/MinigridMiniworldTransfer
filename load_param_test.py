from datetime import datetime
from pdb import set_trace
from time import time

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file

from miniworld_grid_env import MiniworldGridEnv
from miniworld_grid_test import GridObsWrapper
from transfer_extractor import GridEnvExtractor


def main():
    data, params, pytorch_variables = load_from_zip_file(
        "models/ppo/minigrid_gotobox_20230430-191010/iter_300000_steps.zip",
        device="auto",
        custom_objects=None,
        print_system_info=False,
    )

    # Create time stamp of experiment
    stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")

    policy_kwargs = dict(features_extractor_class=GridEnvExtractor)

    env = MiniworldGridEnv()
    env = GridObsWrapper(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=1e5,
        save_path=f"./models/ppo/miniworld_grid_transfer_{stamp}/",
        name_prefix="iter",
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/ppo/miniworld_grid_transfer_tensorboard/",
    )

    """
    (features_extractor): GridEnvExtractor(
        (extractors): ModuleDict(
        (door_color): Linear(in_features=6, out_features=32, bias=True)
        (image): Sequential(
            (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
            (1): ReLU()
            (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
            (3): ReLU()
            (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
            (5): ReLU()
            (6): Flatten(start_dim=1, end_dim=-1)
            (7): Linear(in_features=1536, out_features=64, bias=True)
            (8): ReLU()
        )
        )
    )
    """

    model.policy.features_extractor.extractors.door_color.weight = nn.Parameter(
        params["policy"]["features_extractor.extractors.box_color.weight"]
    )
    model.policy.features_extractor.extractors.door_color.bias = nn.Parameter(
        params["policy"]["features_extractor.extractors.box_color.bias"]
    )

    """
    (pi_features_extractor): GridEnvExtractor(
        (extractors): ModuleDict(
        (door_color): Linear(in_features=6, out_features=32, bias=True)
        (image): Sequential(
            (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
            (1): ReLU()
            (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
            (3): ReLU()
            (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
            (5): ReLU()
            (6): Flatten(start_dim=1, end_dim=-1)
            (7): Linear(in_features=1536, out_features=64, bias=True)
            (8): ReLU()
        )
        )
    )
    """

    model.policy.pi_features_extractor.extractors.door_color.weight = nn.Parameter(
        params["policy"]["pi_features_extractor.extractors.box_color.weight"]
    )
    model.policy.pi_features_extractor.extractors.door_color.bias = nn.Parameter(
        params["policy"]["pi_features_extractor.extractors.box_color.bias"]
    )

    """
    (vf_features_extractor): GridEnvExtractor(
        (extractors): ModuleDict(
        (door_color): Linear(in_features=6, out_features=32, bias=True)
        (image): Sequential(
            (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
            (1): ReLU()
            (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
            (3): ReLU()
            (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
            (5): ReLU()
            (6): Flatten(start_dim=1, end_dim=-1)
            (7): Linear(in_features=1536, out_features=64, bias=True)
            (8): ReLU()
        )
        )
    )
    """

    model.policy.vf_features_extractor.extractors.door_color.weight = nn.Parameter(
        params["policy"]["vf_features_extractor.extractors.box_color.weight"]
    )
    model.policy.vf_features_extractor.extractors.door_color.bias = nn.Parameter(
        params["policy"]["vf_features_extractor.extractors.box_color.bias"]
    )

    """
    (mlp_extractor): MlpExtractor(
        (policy_net): Sequential(
        (0): Linear(in_features=96, out_features=64, bias=True)
        (1): Tanh()
        (2): Linear(in_features=64, out_features=64, bias=True)
        (3): Tanh()
        )
        (value_net): Sequential(
        (0): Linear(in_features=96, out_features=64, bias=True)
        (1): Tanh()
        (2): Linear(in_features=64, out_features=64, bias=True)
        (3): Tanh()
        )
    )
    """

    for i in [0, 2]:
        model.policy.mlp_extractor.policy_net[i].weight = nn.Parameter(
            params["policy"][f"mlp_extractor.policy_net.{i}.weight"]
        )
        model.policy.mlp_extractor.policy_net[i].bias = nn.Parameter(
            params["policy"][f"mlp_extractor.policy_net.{i}.bias"]
        )
        model.policy.mlp_extractor.value_net[i].weight = nn.Parameter(
            params["policy"][f"mlp_extractor.value_net.{i}.weight"]
        )
        model.policy.mlp_extractor.value_net[i].bias = nn.Parameter(
            params["policy"][f"mlp_extractor.value_net.{i}.bias"]
        )

    """
    (action_net): Linear(in_features=64, out_features=3, bias=True)
    """

    model.policy.action_net.weight = nn.Parameter(params["policy"]["action_net.weight"])
    model.policy.action_net.bias = nn.Parameter(params["policy"]["action_net.bias"])

    """
    (value_net): Linear(in_features=64, out_features=1, bias=True)
    """

    model.policy.value_net.weight = nn.Parameter(params["policy"]["value_net.weight"])
    model.policy.value_net.bias = nn.Parameter(params["policy"]["value_net.bias"])

    model.learn(
        2e6,
        tb_log_name=f"{stamp}",
        callback=checkpoint_callback,
    )


if __name__ == "__main__":
    main()
