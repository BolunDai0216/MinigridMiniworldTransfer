import fnmatch
import os

import numpy as np
from scipy import integrate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_files(path, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def find_folders(root_dir, folder_pattern):
    matching_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if fnmatch.fnmatch(dirname, folder_pattern):
                matching_folders.append(dirname)
    return matching_folders


def compute_auc(tensorboard_log_dir):
    # Initialize an EventAccumulator and load the event file (TensorBoard log file)
    event_acc = EventAccumulator(tensorboard_log_dir)
    event_acc.Reload()

    # Get the reward data
    reward_data = event_acc.Scalars("rollout/ep_rew_mean")

    # Extract the step numbers and corresponding rewards
    _steps = [entry.step for entry in reward_data]
    _rewards = [entry.value for entry in reward_data]

    # normalize and numpify the steps and rewards
    steps = np.array(_steps) / _steps[-1]
    rewards = np.array(_rewards)

    # compute area-under-curve (AUC)
    area = integrate.trapz(rewards, steps)

    return area


def compute_average_areas(tensorboard_logs_path, prefix="./logs/ppo/"):
    # Use the function to find all files with 'events.out.tfevents' in their names
    event_folders = find_folders(prefix, tensorboard_logs_path)
    event_files = []
    for folder in event_folders:
        event_files.extend(
            find_files(prefix + folder + "/", "*events.out.tfevents*")
        )

    areas = []

    for filename in event_files:
        try:
            areas.append(
                {
                    "filename": filename,
                    "area": compute_auc(filename),
                }
            )
        except:
            continue

    area_arr = np.array([d["area"] for d in areas])

    return areas, area_arr


def main():
    _, _areas_arr = compute_average_areas("miniworld_gotoobj_tensorboard", prefix="./logs/ppo/")
    base_auc = _areas_arr.mean()

    name_list = [
        "mission",
        "mission_freeze",
        "actor",
        "actor_freeze",
        "critic",
        "critic_freeze",
        "mission_actor",
        "mission_actor_freeze",
        "mission_critic",
        "mission_critic_freeze",
        "mission_actor_critic",
        "mission_actor_critic_freeze",
    ]

    np.set_printoptions(precision=2)

    for name in name_list:
        transfer_areas, areas_arr = compute_average_areas(
            f"miniworld_gotoobj_{name}_transfer_*_tensorboard"
        )
        progress = (areas_arr - base_auc) / base_auc
        print(
            f"{name} transfer: {progress.mean() * 100:.3f} ± {progress.std() * 100:.3f}"
        )


if __name__ == "__main__":
    main()
