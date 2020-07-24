import gym
from gym import wrappers
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    all_steps = []
    all_rewards = []
    for i in range(1000):
        env = gym.make("CartPole-v0")
        total_reward = 0.0
        total_steps = 0
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            all_rewards.append(total_reward)
            total_steps += 1
            all_steps.append(i)
            if done:
                break
    
    plt.plot(all_rewards, all_steps, color='red')
    plt.title('some plot stuff idk')
    plt.xlabel('total_reward')
    plt.ylabel('total_steps')
    plt.grid(True)
    plt.show()
