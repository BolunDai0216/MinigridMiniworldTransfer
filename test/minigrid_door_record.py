import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from minigrid_door_test import DoorEnvExtractor, DoorObsWrapper


def main():
    policy_kwargs = dict(features_extractor_class=DoorEnvExtractor)

    video_folder = "logs/videos/"
    video_length = 500

    env = gym.make("MiniGrid-GoToDoor-8x8-v0", render_mode="rgb_array")
    env = DoorObsWrapper(env)

    # Create a dummy vector environment
    vec_env = DummyVecEnv([lambda: env])

    # Create PPO policy and load weights
    ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    ppo = ppo.load("models/ppo/minigrid_door_20230418-174509")

    # Record the video starting at the first step
    vec_env = VecVideoRecorder(
        vec_env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="MiniGrid-GoToDoor-8x8-v0",
    )

    obs = vec_env.reset()
    rewards = 0

    for i in range(2000):
        action, _state = ppo.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        rewards += reward

        if done:
            obs = vec_env.reset()
            print(f"Test reward: {rewards}")
            rewards = 0
            break

    vec_env.close()


if __name__ == "__main__":
    main()
