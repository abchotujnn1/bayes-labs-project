import torch
from torch import nn, optim
from dataset.clr_data import clearance
from model.sgc import SGC_network
from sklearn.metrics import r2_score
from torch_geometric.data import DataLoader

path="E:/ind content/bayes_labs_project/deepchem_data/Dataset_chembl_clearcaco.txt"
dataset=clearance(path)
print(len(dataset))
d_train=dataset[:int((len(dataset))*0.9)]
d_test =dataset[int((len(dataset))*0.9):]

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class Train(object):
    def __init__(self, model, parameters, epochs, lr):
        # self.model = model(*parameters).to('cuda')
        self.model = model(**parameters)
        self.epoch = epochs
        self.criterion = RMSELoss()
        self.lr = lr
        # self.batch_size=batch_size
    def __call__(self, *input):
       train_loader = DataLoader(input[0], batch_size=32, shuffle=True)

       optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
       for epoch in range(self.epoch):
          l = 0
          count = 0
          for data in train_loader:
              optimizer.zero_grad()
              # data = data.to('cuda')
              self.model = self.model.train()
              output = self.model(data)
              loss = self.criterion(output, data.y)
              loss.backward()
              optimizer.step()
              l += loss.item()
              count += 1
          if (epoch % 100 == 0):
              print(l / count)

          torch.save(self.model, 'E:/ind content/pycharm/office_pro/demo_project/Training/clr.pt')
       test_loader = DataLoader(input[1], batch_size=1, shuffle=False)
       y_hat = [0.0] * len(test_loader)
       y = [0.0] * len(test_loader)
       model = torch.load('E:/ind content/pycharm/office_pro/demo_project/Training/clr.pt')
       model = model.eval()
       count1 = 0
       for data in test_loader:
           out=model(data)
           # out = model(data.to('cuda'))
           y_hat[count1] += out.item()
           y[count1] += float(data.y.view(1))
           count1 += 1
       r_score = r2_score(y, y_hat)
       print("r_score:",r_score)
       return r_score

if __name__=="__main__":
    params = {'in_channel': 40,
              'hid1': 128,
              'hid2': 256,
              'hid3': 128,
              'lin1': 512,
              'lin2': 128,
              'out': 1,
              'drop': 0.5,
              'K': 2}
training = Train(SGC_network, params, 1, 0.001)
training(d_train,d_test)
