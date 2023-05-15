import fnmatch
import os
import pprint

import numpy as np
from scipy import integrate
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator


def find_files(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                # Compute the relative path to the file from the provided path
                relative_path = os.path.relpath(os.path.join(root, name), path)
                result.append(relative_path)
    return result


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


def main():
    tensorboard_logs_path = "logs/ppo/miniworld_gotoobj_tensorboard/"
    
    # Use the function to find all files with 'events.out.tfevents' in their names
    event_files = find_files('*events.out.tfevents*', tensorboard_logs_path)

    areas = []

    for filename in event_files:
        try:
            areas.append({
                "filename": tensorboard_logs_path + filename,
                "area": compute_auc(tensorboard_logs_path + filename)
            })
        except:
            continue
    
    # Create a pretty printer
    pp = pprint.PrettyPrinter(indent=4)

    # Use it to print the dictionary
    pp.pprint(areas)


if __name__ == "__main__":
    main()
