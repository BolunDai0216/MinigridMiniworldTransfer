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


def compute_average_areas(tensorboard_logs_path):
    # Use the function to find all files with 'events.out.tfevents' in their names
    event_folders = find_folders("./logs/ppo/", tensorboard_logs_path)
    event_files = []
    for folder in event_folders:
        event_files.extend(
            find_files("./logs/ppo/" + folder + "/", "*events.out.tfevents*")
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
    base_auc = compute_auc(
        "logs/ppo/miniworld_gotoobj_tensorboard/20230508-113549_1/events.out.tfevents.1683560149.lambda-vector.3171936.6"
    )

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

    for name in name_list:
        transfer_areas, areas_arr = compute_average_areas(
            f"miniworld_gotoobj_{name}_transfer_*_tensorboard"
        )

        print(f"{name} transfer: {(areas_arr.mean() - base_auc) / base_auc}")


if __name__ == "__main__":
    main()
