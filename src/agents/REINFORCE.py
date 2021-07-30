import torch
import torch.nn as nn
import src
from src.common.set_gpu import set_gpu
from torch.distributions.categorical import Categorical

class REINFORCE(nn.Module):

    def __init__(self,
                policy: nn.Module,
                gamma: float = 1.0,
                lr: float = 0.0002):
      super(REINFORCE, self).__init__()
      self.policy = policy
      self.opt = torch.optim.Adam(self.policy.parameters(), lr = lr)
      self.gamma = gamma
      self._eps = 1e-25 # to prevent numbers getting to small that makes precision inaccurate
      self.device = set_gpu()

    def get_action(self, state):
      with torch.no_grad(): # avoid computing grad since we don't need while sampling action from policy
        logits = self.policy(state)
        distribution = Categorical(logits=logits) # equavlent softmax layer 
        action = distribution.sample()

      return action

    def reverse_episode(self, episode):
        states, actions, rewards = episode

        # reversing inputs
        states = states.flip(dims=[0]).to(self.device)
        actions = actions.flip(dims=[0]).to(self.device)
        rewards = rewards.flip(dims=[0]).to(self.device)
        return states, actions, rewards

    def update(self, states, actions, returns, use_norm = False):

      if use_norm: # normalizing returns
        returns = (returns - returns.mean()) / (returns.std() + self._eps)
      
      distribution = Categorical(logits = self.policy(states)) #generate categorical distribution using output of policy network(logits)
      prob = distribution.probs[range(states.shape[0]),actions] # calculating pi(a|s) for actions from episodes

      loss = (- torch.log(prob.to(self.device) + self._eps) * returns.squeeze()).to(self.device) # need to be negative since optimizer in pytorch trying 
      # to do gradient descent  to minimize loss, but we want to maximize the objective function using gradient ascent to do gradient descent 
                                             
      self.opt.zero_grad() # first need to clean out the grads
      loss = loss.mean()
      loss.backward() # computing gradient
      self.opt.step() # updating theta

