
import torch
from torch import nn

from torch.nn import Linear, ReLU,Tanh
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F



class GAT(torch.nn.Module):
  def __init__(self,input_dim,hidden_dim1, hidden_dim, output_dim, num_layers,):
    super(GAT,self).__init__()
    torch.manual_seed(1234)
    self.convs = None
    def get_in_channels(idx):
        if idx>1:
          return hidden_dim
        elif idx==1:
          return hidden_dim1
        else:
          return input_dim 

    def get_out_channels(idx):
      if idx==0:
        return hidden_dim1
      elif idx>0 and idx<=num_layers-1:
        return hidden_dim
      else:
        return output_dim
    self.convs = torch.nn.ModuleList([
          GATConv(in_channels=get_in_channels(i), out_channels=get_out_channels(i))
          for i in range(num_layers)
      ])

    self.batch_norms = torch.nn.ModuleList([nn.BatchNorm1d(get_out_channels(i)) for i in range(num_layers)])
    self.linear1 = torch.nn.Linear(hidden_dim,hidden_dim)
    self.linear = torch.nn.Linear(hidden_dim, output_dim)
  def forward(self, x, edge_index):
    
    for i, (gat,batch_norm) in enumerate(zip(self.convs,self.batch_norms)):
      # if i == 0:
      #   x = torch.cat([x, edge_attr], dim=-1)

      # Perform message passing with concatenated features
      x = gat(x, edge_index)
      x = batch_norm(x)

      x = F.relu(x)
    x=self.linear1(x)
    x=F.relu(x)
    x=self.linear1(x)
    x = F.relu(x)
    out = self.linear(x)
    return out
