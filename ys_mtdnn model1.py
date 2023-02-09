
#model 1 is comparing the output layer (4 neuron) against 1 toxicity y value



import numpy as np
import pandas as pd
#from numpy import mean
#from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.model_selection import KFold
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



#for mtdnn, we train all
T1 = np.load ("npy inputs\lc50dm_train_x_ssl.npy")
T1 = np.float32(T1)
T1 = torch.from_numpy(T1)
n1 = len(T1)

T2 = np.load ("npy inputs\LD50_train_x_ssl.npy")
T2 = np.float32(T2)
T2 = torch.from_numpy(T2)
n2 = len(T2)


T3 = np.load ("npy inputs\IGC50_train_x_ssl.npy")
T3 = np.float32(T3)
T3 = torch.from_numpy(T3)
n3 = len(T3)


T4 = np.load ("npy inputs\lc50_train_x_ssl.npy")
T4 = np.float32(T4)
T4 = torch.from_numpy(T4)
n4 = len(T4)

train_X = torch.cat((T1,T2,T3,T4))

train_y1 = pd.read_csv("toxicity dataset (csv and mol)\lc50dm\lc50dm_train.csv")
train_y1 = train_y1['label']
train_y1 = np.float32(train_y1)
train_y1 = torch.from_numpy(train_y1)

train_y2 = pd.read_csv("toxicity dataset (csv and mol)\ld50\ld50_train.csv")
train_y2 = train_y2['label']
train_y2 = np.float32(train_y2)
train_y2 = torch.from_numpy(train_y2)

train_y3 = pd.read_csv("toxicity dataset (csv and mol)\igc50\igc50_train.csv")
train_y3 = train_y3['label']
train_y3 = np.float32(train_y3)
train_y3 = torch.from_numpy(train_y3)

train_y4 = pd.read_csv("toxicity dataset (csv and mol)\lc50\lc50_train.csv")
train_y4 = train_y4['label']
train_y4 = np.float32(train_y4)
train_y4 = torch.from_numpy(train_y4)

train_y = torch.cat((train_y1, train_y2, train_y3, train_y4))

class Build_Data(Dataset):    
    # Constructor
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = self.x.shape[0]        
    # Getting the data
    def __getitem__(self, index):    
        return self.x[index], self.y[index]    
    # Getting length of the data
    def __len__(self):
        return self.len
 
# Creating DataLoader object
train_dataset = Build_Data(train_X, train_y)
train_data_iter = DataLoader(dataset = train_dataset, batch_size = 8)


class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()    #512 (1024 512 512 512) 1
       self.input_layer = torch.nn.Linear(512, 1024)
       self.layer1 = torch.nn.Linear(1024, 512)
       self.layer2 = torch.nn.Linear(512, 512)
       self.layer3 = torch.nn.Linear(512, 512)
       self.output_layer = torch.nn.Linear(512,4)
       
       
   def forward(self, x):
       x = torch.relu(self.input_layer(x))
       x = torch.relu(self.layer1(x))
       x = torch.relu(self.layer2(x))
       x = torch.relu(self.layer3(x))
       x = self.output_layer(x)      
       return x

net = Net()
print("This is Net:", net)

#criterion = nn.NLLLoss()
criterion = nn.MSELoss()

optimizer1 = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
optimizer2 = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

'''
#to check before and after
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    '''


num_epoch = 10 #2000
count = 0
loss_list = []

for epoch in range(num_epoch):
    count+=1
    if epoch <= 1000:
        optimizer = optimizer1
    else:
        optimizer = optimizer2
    for X,y in train_data_iter:
        #print("X:",X) #x values in batches of 8, one x value is 512 variables
        #print("y:",y) #y values in batches of 8, one y value is just 1 value, the toxicity
        optimizer.zero_grad()
        out=net(X)
        target = y
        target = target.view(-1,1)
        #print("out:",out)
        loss = criterion(out,target)
        loss.backward()
        
        print("count:",count)
        #print("count:", count) #expect TRAIN_DATA_ITER*num_epoch
        #print("loss:", loss)
        loss_list.append(loss.item())
        optimizer.step()
        #if count in range(0,5):
        #    print("count:",count)
        #    print("out:", out)
        #    print("target:",target)
        
step = np.linspace(0, num_epoch, len(loss_list))

fig, ax = plt.subplots(figsize=(8,5))
epochs = range(1,count)
plt.plot(step, np.array(loss_list))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show() 

PATH = 'MTDNN_ssl_statedict_MSE1.pt'
torch.save(net.state_dict(), PATH)
PATH1 = "MTDNN_ssl_MSE1.pt"
torch.save(net, PATH1)
