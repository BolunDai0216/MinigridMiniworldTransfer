import gymnasium as gym
import miniworld
from stable_baselines3 import PPO

from miniworld_gotoobj_env import MiniworldGoToObjEnv
from miniworld_gotoobj_train import GoToObjEnvExtractor, GoToObjObsWrapper
from stable_baselines3.common.save_util import load_from_zip_file


def main():
    data1, params1, pytorch_variables1 = load_from_zip_file(
        "models/ppo/miniworld_gotoobj_actor_freeze_transfer_20230517-225219/iter_100000_steps.zip",
        device="auto",
        custom_objects=None,
        print_system_info=False,
    )

    data2, params2, pytorch_variables2 = load_from_zip_file(
        "models/ppo/miniworld_gotoobj_actor_freeze_transfer_20230517-225219/iter_200000_steps.zip",
        device="auto",
        custom_objects=None,
        print_system_info=False,
    )

    change = (
        params1["policy"]["action_net.weight"] - params2["policy"]["action_net.weight"]
    )

    breakpoint()


if __name__ == "__main__":
    main()
