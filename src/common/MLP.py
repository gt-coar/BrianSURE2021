import torch
import torch.nn as nn
from src.common.set_gpu import set_gpu

class mlp(nn.Module):

    def __init__(self,
                input_dim: int,
                output_dim: int, 
                num_nureons: list,
                hidden_act: str ='ReLU', 
                output_act: str = 'Identity'):
      super(mlp,self).__init__()

      self.input_dim = input_dim
      self.input_dims = [input_dim] + num_nureons
      self.output_dim = output_dim
      self.output_dims = num_nureons + [output_dim]
      self.num_nureons = num_nureons
      self.device = set_gpu()
      self.hidden_act = getattr(nn, hidden_act)().to(self.device)
      self.output_act = getattr(nn, output_act)().to(self.device)
      self.layers = nn.ModuleList().to(self.device)
    

      for ind , (in_d, out_d) in enumerate(zip(self.input_dims, self.output_dims)):
        self.layers.append(nn.Linear(in_d, out_d).to(self.device))
        last = False
        if ind == (len(self.input_dims) - 1): # checking if it is last layer
          last = True
        else:
          last = False 

        if last: # Last layer append output activation function
          self.layers.append(self.output_act)
        else: # For Hidden Layer append hidden activation function
          self.layers.append(self.hidden_act)

    def forward(self, x): 
      # Performs forward propagation
      for layer in self.layers:
        x = layer(x)
      return x.to(self.device)
   