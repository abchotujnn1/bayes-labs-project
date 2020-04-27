import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SGConv
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from data import*
from model import*
class Test(object):
    def __init__(self, model):
        self.model= model.eval()
        # self.model=self.model.to("cpu")
        self.sigmoid=nn.Sigmoid()

    def __call__(self,test_data):
        # y=[]
        # y_hat=[]
        count=0
        correct=np.array([0]*306)
        for data in test_data:
            output=self.model(data)
            output=output.data
            output=self.sigmoid(output)
            y_hat=torch.tensor([1 if out>=0.5 else 0 for out in output.squeeze().data])
            y=data.y.squeeze().data
            correct+=np.array(list(map(int,(y==y_hat))))
            count+=1
        print(correct)

        accuracy=correct/count
        print(accuracy)
        return 0

if __name__=="__main__":
    path = "E:/ind content/bayes_labs_project/deepchem_data/biological_pathway_prediction/smile_pathway.csv"
    dataset = Bio_Pathway_Dataset(path)
    dataset = dataset[int(len(dataset)*0.9):]
    test_data= DataLoader(dataset, batch_size=1, shuffle=True)
    model_path = 'biological_pathway.pt'
    model = torch.load(model_path, map_location=torch.device("cpu"))
    print(model)
    test = Test(model)
    test(test_data)





