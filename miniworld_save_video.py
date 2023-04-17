import gymnasium as gym
import miniworld
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


def main():
    video_folder = "logs/videos/"
    video_length = 1000

    env = gym.make("MiniWorld-OneRoom-v0", render_mode="rgb_array")

    # Create a dummy vector environment
    vec_env = DummyVecEnv([lambda: env])

    # Create PPO policy and load weights
    ppo = PPO("CnnPolicy", env, verbose=1)
    ppo = ppo.load("models/ppo/miniworld_oneroom")

    # Record the video starting at the first step
    vec_env = VecVideoRecorder(
        vec_env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="MiniWorld-OneRoom-v0",
    )

    obs = vec_env.reset()
    rewards = 0

    for i in range(2000):
        action, _state = ppo.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        rewards += reward

        if done:
            break

    print(f"Test reward: {rewards}")

    vec_env.close()


if __name__ == "__main__":
    main()
