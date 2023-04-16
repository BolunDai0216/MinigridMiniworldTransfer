import gymnasium as gym
from pdb import set_trace
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo


def main(train=False):
    if train:
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=2e5)
        model.save("models/ppo/models")

    # Create Environement
    env = gym.make("CartPole-v1")

    # Load Model
    ppo = PPO("MlpPolicy", env, verbose=1)
    ppo.load("models/ppo/models")

    obs, info = env.reset()
    rewards = 0

    for i in range(1000):
        action, _state = ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward

        if terminated or truncated:
            print(f"Test reward: {rewards}")
            obs, info = env.reset()
            rewards = 0
            continue

    env.close()


if __name__ == "__main__":
    main()
