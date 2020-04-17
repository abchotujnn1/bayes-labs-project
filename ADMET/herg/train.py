import torch
from torch import nn
from model_hERG import*
from data_hERG import*
from torch_geometric.data import DataLoader
path="E:/ind content/bayes_labs_project/deepchem_data/HERG_DATASET\herg_data.csv"
dataset=NumbersDataset(path)
d_train=dataset[:3072]
loader = DataLoader(d_train, batch_size=10, shuffle=True)
model=GCN_network()
#############LOSS_FUNCN##################
optimizer = optim.Adam(model.parameters(),lr=0.0005)
criterion = nn.BCELoss()
##############################TRAINING#######################
for epoch in range(1000):
    count=0
    l=0
    for data in loader:
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,data.y)
        loss.backward()
        optimizer.step()
        l+=loss.item()
        count+=1
    if(epoch%10==0):
        print(l/count)

torch.save(model,'model_hERG.pt')
