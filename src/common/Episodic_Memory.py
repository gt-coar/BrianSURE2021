import torch
from src.common.set_gpu import set_gpu
from collections import deque

class Trajectory:

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.states = list()
        self.actions = list()
        self.rewards = list()
        self.next_states = list()
        self.dones = list()
        self.length = 0
        self.returns = None
        self._discounted = False
        self.device = set_gpu()

    def push(self, state, action, reward, next_state, done):
        if done and self._discounted:
            raise RuntimeError("Done occured twice. Episode currupted")

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.length += 1

        if done and not self._discounted:
            # compute returns when done
            self.compute_return()

    def compute_return(self):
        rewards = self.rewards
        returns = list()

        g = torch.tensor(0).to(self.device)
        # iterating returns in reverse order
        for r in rewards[::-1]:
            g = r + self.gamma * g
            returns.insert(0, g)
        self.returns = returns
        self._discounted = True

    def get_samples(self):
        return self.states, self.actions, self.rewards, self.next_states, self.dones, self.returns


class EpisodicMemory:

    def __init__(self, max_size: int, gamma: float):
        self.max_size = max_size  # maximum number of trajectories
        self.gamma = gamma
        self.trajectories = deque(maxlen=max_size)
        self._trajectory = Trajectory(gamma=gamma)
        self.device = set_gpu()

    def push(self, state, action, reward, next_state, done):
        self._trajectory.push(state, action, reward, next_state, done)
        if done:
            self.trajectories.append(self._trajectory)
            self._trajectory = Trajectory(gamma=self.gamma)

    def reset(self):
        self.trajectories.clear()
        self._trajectory = Trajectory(gamma=self.gamma)

    def get_samples(self):
        states, actions, rewards, next_states, dones, returns = [], [], [], [], [], []
        while self.trajectories:
            traj = self.trajectories.pop() # Pop one trajectory from memory
            s, a, r, ns, done, g = traj.get_samples()
            states.append(torch.cat(s, dim=0).to(self.device))
            actions.append(torch.cat(a, dim=0).to(self.device))
            rewards.append(torch.cat(r, dim=0).to(self.device))
            next_states.append(torch.cat(ns, dim=0).to(self.device))
            dones.append(torch.cat(done, dim=0).to(self.device))
            returns.append(torch.cat(g, dim=0).to(self.device))

        states = torch.cat(states, dim=0).to(self.device)
        actions = torch.cat(actions, dim=0).to(self.device)
        rewards = torch.cat(rewards, dim=0).to(self.device)
        next_states = torch.cat(next_states, dim=0).to(self.device)
        dones = torch.cat(dones, dim=0).to(self.device)
        returns = torch.cat(returns, dim=0).to(self.device)

        return states, actions, rewards, next_states, dones, returns
 