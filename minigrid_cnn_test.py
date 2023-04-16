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
    model.learn(1e6)
    model.save("models/ppo/minigrid_empty")


if __name__ == "__main__":
    main()
