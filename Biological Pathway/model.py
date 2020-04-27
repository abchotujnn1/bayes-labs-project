import torch
import torch.nn as nn
from torch_geometric.nn import SGConv
from torch_geometric.nn import GATConv
from torch.nn import functional as F

class SGC_class_network(torch.nn.Module):
    def __init__(self,in_channel, hid1, hid2, hid3, lin1,lin2, out, drop, K):
        super(SGC_class_network, self).__init__()
        self.drop=drop
        self.conv1=SGConv(in_channel,hid1,K)
        self.conv2=SGConv(hid1,hid2,K)
        self.conv3=GATConv(hid2, hid3)

        self.l1=nn.Linear(hid3*40, lin1)
        self.l2=nn.Linear(lin1, lin2)
        self.l3=nn.Linear(lin2, out)
        self.r=nn.LeakyReLU(0.1)
    def forward(self, input):
        x=input.x
        edge_index =input.edge_index
        x=self.r(self.conv1(x, edge_index))
        x=F.dropout(x, self.drop, training=self.training)
        x=self.r(self.conv2(x, edge_index))
        x= F.dropout(x, self.drop, training=self.training)
        x=self.r(self.conv3(x, edge_index))
        x=F.dropout(x, self.drop, training=self.training)

        x=x.view(int(len(x)/40),-1)

        x=self.r(self.l1(x))
        x=F.dropout(x, self.drop, training=self.training)
        x=self.r(self.l2(x))
        x=F.dropout(x, self.drop, training=self.training)
        x=self.r(self.l3(x))
        return x

if __name__ =="__main__":
    hparams={'in_channel':40,
              'hid1': 128,
              'hid2': 256,
              'hid3': 128,
              'lin1': 1024,                                        #'lin1': 512,
              'lin2': 512,                                          #'lin2': 128,
              'out': 306,
              'drop': 0.5,
              'K': 2
             }
    model=SGC_class_network(**hparams)
    print(model)


