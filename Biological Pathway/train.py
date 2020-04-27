import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader
from model import*
from data import*
import warnings
from warnings import filterwarnings("ignore")
class Train(object):
    def __init__(self, model, learning_rate, epochs):
        self.model=model.train()
        self.lr=learning_rate
        self.epochs=epochs
        self.criterion=nn.BCEWithLogitsLoss()
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.lr)

    def __call__(self, train_loader):
        for _ in range(self.epochs):
            loss=0
            count=0
            for data in train_loader:
                # print(data)
                y=data.y
                self.optimizer.zero_grad()
                output=self.model(data)
                loss=self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                loss+=loss.item()
                count+=1
            print(loss/count)
        torch.save(self.model, 'biological_pathway.pt')
        return 0

if __name__ == "__main__":
    path = "E:/ind content/bayes_labs_project/deepchem_data/biological_pathway_prediction/smile_pathway.csv"
    hparams = {'in_channel': 40,
               'hid1': 128,
               'hid2': 256,
               'hid3': 128,
               'lin1': 1024,  # 'lin1': 512,
               'lin2': 512,   # 'lin2': 128,
               'out': 306,
               'drop': 0.5,
               'K': 2
               }
    params = {'model': SGC_class_network(**hparams),
    'learning_rate': 0.001,
    'epochs': 1}

    dataset = Bio_Pathway_Dataset(path)
    print(dataset[0])
    train_data = DataLoader(dataset, batch_size=64, shuffle=True)
    training = Train(**params)
    training(train_data)
