import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import SGConv
class SGC_network(torch.nn.Module):
    def __init__(self,in_channel, hid1, hid2, hid3, lin1,lin2,out, drop, K):
        super(SGC_network,self).__init__()
        self.drop=drop
        self.conv1=SGConv(in_channel,hid1,K)
        self.conv2=SGConv(hid1,hid2,K)
        self.conv3=GATConv(hid2,hid3)

        self.l1=nn.Linear(40*hid3,lin1)
        self.l2=nn.Linear(lin1,lin2)
        self.l3=nn.Linear(lin2,out)
        self.l=nn.LeakyReLU(0.1)
    def forward(self,data):
        x=data.x
        edge_index=data.edge_index
        x=self.l(self.conv1(x,edge_index))
        x = F.dropout(x, self.drop, training=self.training)
        x=self.l(self.conv2(x,edge_index))
        x=self.l(self.conv3(x,edge_index))
        self.grad_value = x.clone()
        x=x.view(int(len(x)/40),-1)
        x=self.l(self.l1(x))
        x=F.dropout(x,self.drop,training=self.training)
        x=self.l(self.l2(x))
        x=F.dropout(x,self.drop, training=self.training)
        x=self.l3(x)
        return x
    def cam(self):
        return self.l1.weight.data,self.grad_value


# model=GCN_network(0.5)
# print(model)
