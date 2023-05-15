import numpy as np
from scipy import integrate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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
    tensorboard_log_dir = "logs/ppo/miniworld_gotoobj_tensorboard/20230508-113549_1/events.out.tfevents.1683560149.lambda-vector.3171936.6"
    area = compute_auc(tensorboard_log_dir)

    print(area)


if __name__ == "__main__":
    main()
