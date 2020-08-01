from typing import List

from collections import deque
import random
import math

import numpy as np

import torch
import torch.nn

from tqdm import trange

import gym
from gym import wrappers
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("NVIDIA")
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = torch.device(dev)  

N_STATES = 2
N_ACTIONS = 2  # Left, Right
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MEMORY_SIZE = int(1e3)
STARTUP_SIZE = 100
TOTAL_RUNTIME = int(MEMORY_SIZE*2)
EPSILON_DECAY = 0.998
GAMMA = 0.9999

assert STARTUP_SIZE < TOTAL_RUNTIME

## ======= States =================
def angle_to_vector(pole_angle_rad: float, n: int) -> np.ndarray:
    bins = np.linspace(-np.pi/2, np.pi/2, n+1)[1:-1]
    idx = np.digitize(pole_angle_rad, bins)
    out = np.zeros(n, dtype="float32")
    out[idx] = 1
    return out

def test_angle_to_vector():
    assert (angle_to_vector(0, 2) == [0, 1]).all()
    assert (angle_to_vector(math.radians(-15), 2) == [1, 0]).all()
    assert (angle_to_vector(math.radians(15), 2) == [0, 1]).all()

## ======= Q Network ==============
class QNetwork(torch.nn.Module):
    inputlayer: torch.nn.Linear
    hidden: List[torch.nn.Linear]
    output: torch.nn.Linear

    def __init__(self):
        super(QNetwork, self).__init__()
        H = [10, 100, 10]
        self.hidden = []
        self.inputlayer = torch.nn.Linear(N_STATES, H[0]).to(device)
        for h0, h1 in zip(H, H[1:]):
            self.hidden.append(torch.nn.Linear(h0, h1).to(device))

        self.output = torch.nn.Linear(H[-1], N_ACTIONS).to(device)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inputlayer(x).clamp(min=0)
        for h in self.hidden:
            y = h(y).clamp(min=0)
        y = self.output(y)
        return y

    def fit_once(self, state: torch.Tensor, reward: torch.Tensor) -> None:
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = self(state)

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
        all_state_vec = []
        all_qvalues = []
        for state, action, reward, state_next, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.argmax(model(torch.from_numpy(angle_to_vector(state_next, N_STATES)).to(device)).cpu().detach().numpy()))

            state_vec = torch.from_numpy(angle_to_vector(state, N_STATES)).to(device)
            all_state_vec.append(state_vec)
            q_values = model(state_vec)
            q_values[action] = q_update
            all_qvalues.append(q_values)
        model.fit_once(torch.stack(all_state_vec), torch.stack(all_qvalues))

## ======= Plotting ==============

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

## ======= Gym ====================

if __name__ == "__main__":
    memory = Memory([], MEMORY_SIZE)
    model = QNetwork()
    epsilon = 1
    all_epsilons = []
    all_steps = []
    all_rewards = []
    for i in trange(TOTAL_RUNTIME):
        env = gym.make("CartPole-v0")
        total_reward = 0.0
        total_steps = 0
        state = env.reset()[2]  # Represents the pole's angle
        while True:
            if i < STARTUP_SIZE:
                action = env.action_space.sample()
            else:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(model(torch.from_numpy(angle_to_vector(state, N_STATES)).to(device)).cpu().detach().numpy())
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[2]
            total_reward += reward
            total_steps += 1
            all_rewards.append(total_reward)
            all_steps.append(i)
            all_epsilons.append(epsilon)
            memory.push(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        if i >= STARTUP_SIZE:
            memory.experience_replay(model)
        epsilon *= EPSILON_DECAY

    MOVING_AVERAGE = 100
    fig, (ax1, ax2) = plt.subplots(2)    
    ax1.plot(all_steps, all_rewards, color='red')
    ax1.plot(all_steps[MOVING_AVERAGE-1:], moving_average(all_rewards, n=MOVING_AVERAGE), color='blue')
    ax1.set_title('some plot stuff idk')
    ax1.set_xlabel('total_reward')
    ax1.set_ylabel('total_steps')
    ax2.plot(all_steps, all_epsilons, color='red')
    ax2.set_title('Epsilon vs Steps')
    ax2.set_xlabel('epsilon')
    ax2.set_ylabel('total_steps')
    plt.show()
