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

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/cartpole_binary')

# GPU
if torch.cuda.is_available():
    print("NVIDIA")
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = torch.device(dev)  

# Randomness control
extra = 0
torch.manual_seed(757423+extra)
np.random.seed(3823823+extra)
random.seed(84873789+extra)

# Constants
## State Space
N_POS_STATES = 3
N_VEL_STATES = 2
N_ANGLE_STATES = 3
N_ANGLE_VEL_STATES = 2
N_STATES = N_POS_STATES + N_VEL_STATES + N_ANGLE_STATES + N_ANGLE_VEL_STATES
N_ACTIONS = 2  # Left, Right

## Model Parameters
H = [100, 1000, 100]
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
EXPLORATION_LEVEL_DECAY = 0.9998  # REF: https://www.desmos.com/calculator/pc8u25qzt3
GAMMA = 0.9999

## Runtime
STARTUP_SIZE = 100
MEMORY_SIZE = int(1e4)
TOTAL_RUNTIME = int(MEMORY_SIZE*2)
RENDER_EVERY_K_RUNS = 100

# Checking Constants
assert STARTUP_SIZE < TOTAL_RUNTIME

## ======= States =================
class State():
    """
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    """
    vel_scale = 1.0
    angle_scale = 1.0

    def __init__(self, statevec: np.ndarray):
        self.position = self.binarize(statevec[0], -4.8, 4.8, N_POS_STATES)
        self.velocity = self.binarize(self.vel_scale * statevec[1], -1, 1, N_VEL_STATES)
        self.angle = self.binarize(statevec[2], -0.418, 0.418, N_ANGLE_STATES) # self.angle_to_vector(statevec[2], N_ANGLE_STATES)
        self.angle_vel = self.binarize(self.angle_scale * statevec[3], -1, 1, N_ANGLE_VEL_STATES)

    def clip(self, value: float, minval: float, maxval: float) -> float:
        return (value - minval) / (maxval - minval)

    @staticmethod
    def binarize(value: float, minval: float, maxval: float, n: int) -> np.ndarray:
        bins = np.linspace(minval, maxval, n+1)[1:-1]
        idx = np.digitize(value, bins)
        out = np.zeros(n, dtype="float32")
        out[idx] = 1
        return out
    
    def as_vector(self):
        out = np.zeros(N_STATES, dtype="float32")
        out[0:N_POS_STATES] = self.position
        w = N_POS_STATES + N_VEL_STATES 
        out[N_POS_STATES:w] = self.velocity
        w2 = w + N_ANGLE_STATES
        out[w:w2] = self.angle
        w3 = w2 + N_ANGLE_VEL_STATES
        out[w2:w3] = self.angle_vel
        return out[np.newaxis, :]

def test_angle_to_vector():
    assert (State.angle_to_vector(0, 2) == [0, 1]).all()
    assert (State.angle_to_vector(math.radians(-15), 2) == [1, 0]).all()
    assert (State.angle_to_vector(math.radians(15), 2) == [0, 1]).all()

## ======= Q Network ==============
class QNetwork(torch.nn.Module):
    inputlayer: torch.nn.Linear
    hidden: List[torch.nn.Linear]
    output: torch.nn.Linear

    def __init__(self):
        super(QNetwork, self).__init__()
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
    def push(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
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
                q_update = (reward + GAMMA * np.argmax(model(torch.from_numpy(state_next.as_vector()).to(device)).cpu().detach().numpy()))
            state_vec = torch.from_numpy(state.as_vector()).to(device)
            all_state_vec.append(state_vec)
            q_values = model(state_vec)
            q_values[0][action] = q_update
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
    exploration_rate = 1
    all_exploration_rates = []
    all_steps = []
    all_rewards = []
    env = gym.make("CartPole-v0")
    for i in trange(TOTAL_RUNTIME):
        total_reward = 0.0
        total_steps = 0
        state = State(env.reset())
        j = 0
        done = False
        while True:
            if i % RENDER_EVERY_K_RUNS == 0:
                env.render()
            if i < STARTUP_SIZE:
                action = env.action_space.sample()
            else:
                if random.random() < exploration_rate:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(model(torch.from_numpy(state.as_vector()).to(device)).cpu().detach().numpy())
            next_state, reward, done_, _ = env.step(action)
            next_state = State(next_state)
            done = done or done_  # Toggled by done_ until we exit the loop
            total_reward += reward
            total_steps += 1
            if done:
                if i % RENDER_EVERY_K_RUNS != 0:
                    break
                else:
                    if j > 100:
                        break
            else:
                all_rewards.append(total_reward)
                all_steps.append(i)
                all_exploration_rates.append(exploration_rate)
                memory.push(state, action, reward, next_state, done)
            state = next_state
            j += 1
        if i >= STARTUP_SIZE:
            memory.experience_replay(model)
        exploration_rate *= EXPLORATION_LEVEL_DECAY

    MOVING_AVERAGE = 100
    fig, (ax1, ax2) = plt.subplots(2)    
    ax1.plot(all_steps, all_rewards, color='red')
    ax1.plot(all_steps[MOVING_AVERAGE-1:], moving_average(all_rewards, n=MOVING_AVERAGE), color='blue')
    ax1.set_title('some plot stuff idk')
    ax1.set_ylabel('total_reward')
    ax1.set_xlabel('total_steps')
    ax2.plot(all_steps, all_exploration_rates, color='red')
    ax2.set_title('Exploration Rate vs Steps')
    ax2.set_ylabel('exploration_rate')
    ax2.set_xlabel('total_steps')
    plt.show()
