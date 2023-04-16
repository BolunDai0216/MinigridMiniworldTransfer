from minigrid_cnn import MinigridCNN
import minigrid
from minigrid.wrappers import ImgObsWrapper, ViewSizeWrapper
import gymnasium as gym
from stable_baselines3 import PPO
from pdb import set_trace


def main():
    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env = gym.make("MiniGrid-Empty-16x16-v0")
    env = ImgObsWrapper(env)

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # model.learn(1e5)

    # model.save("models/ppo/minigrid_empty")
    model.load("models/ppo/minigrid_empty", env=env)

    obs, info = env.reset()
    rewards = 0

    for i in range(2000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward

        if terminated or truncated:
            print(f"Test reward: {rewards}")
            obs, info = env.reset()
            rewards = 0
            continue

    print(f"Test reward: {rewards}")

    env.close()


if __name__ == "__main__":
    main()
