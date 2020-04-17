import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_hERG import*
model=torch.load('model_hERG.pt')
model=model.eval()
path="E:/ind content/bayes_labs_project/deepchem_data/HERG_DATASET\herg_data.csv"
dataset= NumbersDataset(path)
d_test=dataset[3072:]
def cam(linear_wt,logits):
    linear_wt = linear_wt.detach()
    logits = logits.detach()
    # ll=[]
    # j=0
    for i in linear_wt:
        i=i.view(40,48).t()
#       print(i.shape)
        l=torch.matmul(logits,i)
#       print(l.shape)
        l=torch.sum(l,axis=1)
        l=F.relu(l)
        # ll.append(l.numpy())
        return [l.numpy()]



def prediction(model,d_test):
    y_hat=[]
    yy=[]
    for data in d_test:
        output=model(data)
        y_hat.append(output.item())
        yy.append(float(data.y.view(1)))
    return yy,y_hat

def accuracy_measure(y,y_hat):
    y_l=[]
    for i in y_hat:
        if(i>=0.50):
            y_l.append(1.0)
        else:
            y_l.append(0.0)
    F_score=f1_score(y,y_l)
    # ROC=roc_auc_score(y,y_hat)
    cm=confusion_matrix(y,y_l)
    return F_score,cm

y,y_hat=prediction(model,d_test)
f,c=accuracy_measure(y,y_hat)
print(f)
# print(r)
print(c)
