import torch
from torch import nn,optim
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

class GCN_network(torch.nn.Module):
    def __init__(self):
        super(GCN_network,self).__init__()
        self.conv1=GCNConv(58,96)
        self.conv2=GCNConv(96,48)
        self.l1=nn.Linear(40*48,256)
        self.l2=nn.Linear(256,64)
        self.l3=nn.Linear(64,1)

    def forward(self,data):
        x=data.x
        edge_index=data.edge_index
        x=F.dropout(F.relu(self.conv1(x,edge_index)),training=self.training)
        x=F.relu(self.conv2(x,edge_index))
        self.grad_value = x.clone()
        x=x.view(int(len(x)/40),-1)
        x=F.dropout(F.relu(self.l1(x)),training=self.training)
        x=F.dropout(F.relu(self.l2(x)),training=self.training)
        x=self.l3(x)
        x=torch.sigmoid(x)
        return x
    def cam(self):
        return self.l1.weight.data,self.grad_value
# model=GCN_network()
# print(model)

