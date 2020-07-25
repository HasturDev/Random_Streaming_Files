from typing import List

from collections import deque
import random

import numpy as np

import torch
import torch.nn

import gym
from gym import wrappers
import matplotlib.pyplot as plt

N_STATES = 4
N_ACTIONS = 3  # Left, Stop, Right
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MEMORY_SIZE = int(1e6)
EPSILON_DECAY = 0.0001
GAMMA = 0.9999

## ======= States =================
def angle_to_vector(pole_angle: float, n: int) -> torch.tensor:
    bins = np.linspace(-np.pi/2, np.pi/2, n)
    idx = np.digitize(pole_angle, bins)
    out = np.zeros(n, dtype=float)
    out[idx] = 1
    return torch.from_numpy(out)

## ======= Q Network ==============
class QNetwork(torch.nn.Module):
    input: torch.nn.Linear = None
    hidden: List[torch.nn.Linear] = None
    output: torch.nn.Linear = None

    def __init__(self):
        super(QNetwork, self).__init__()
        H = [10, 100, 10]
        self.hidden = []
        self.input = torch.nn.Linear(N_STATES, H[0])
        for h0, h1 in zip(H, H[1:]):
            self.hidden.append(torch.nn.Linear(h0, h1))
        self.output = torch.nn.Linear(H[-1], N_ACTIONS)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x: torch.tensor) -> torch.tensor:
        y = self.input(x).clamp(min=0)
        for h in self.hidden:
            y = h(y).clamp(min=0)
        y = self.output(y)
        return y

    def fit_once(self, state: torch.tensor, reward: torch.tensor) -> None:
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = self.forward(state)

        # Compute and print loss
        loss = self.criterion(y_pred, reward)

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

## ======= Memory Model ===========
class Memory(deque):
    def push(self, state: float, action: int, reward: float, next_state: float, done: bool) -> None:
        super(Memory, self).append((state, action, reward, next_state, done))

    ## ======= Bellman Equation =======
    def experience_replay(self, model):
        if len(self) < BATCH_SIZE:
            return
        batch = random.sample(self, BATCH_SIZE)
        for state, action, reward, state_next, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.amax(model.forward(angle_to_vector(state_next, N_STATES))[0]))
            q_values = model.forward(angle_to_vector(state, N_STATES))
            q_values[0][action] = q_update
            model.fit_once(state, q_values)

## ======= Gym ====================

if __name__ == "__main__":
    memory = Memory(MEMORY_SIZE)
    model = QNetwork()
    epsilon = 1
    all_steps = []
    all_rewards = []
    for i in range(MEMORY_SIZE*100):
        env = gym.make("CartPole-v0")
        total_reward = 0.0
        total_steps = 0
        state = env.reset()
        while True:
            if i < MEMORY_SIZE:
                action = env.action_space.sample()
            else:
                action = env.action_space.sample() if random.random() < epsilon else model(angle_to_vector(state, N_STATES))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            all_rewards.append(total_reward)
            total_steps += 1
            all_steps.append(i)
            memory.push(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        epsilon *= EPSILON_DECAY
    
    plt.plot(all_rewards, all_steps, color='red')
    plt.title('some plot stuff idk')
    plt.xlabel('total_reward')
    plt.ylabel('total_steps')
    plt.grid(True)
    plt.show()
