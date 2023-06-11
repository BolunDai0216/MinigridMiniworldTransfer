import gymnasium as gym
from stable_baselines3 import PPO


def main(train=True):
    if train:
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1e3)
        model.save("models/ppo/cartpole")
        action_net_weights = model.policy.action_net.weight
        del model

    # Create Environement
    env = gym.make("CartPole-v1")

    # Load Model
    ppo = PPO("MlpPolicy", env, verbose=1)
    ppo = ppo.load("models/ppo/cartpole", print_system_info=True)
    loaded_action_net_weights = ppo.policy.action_net.weight

    print(loaded_action_net_weights - action_net_weights)

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
