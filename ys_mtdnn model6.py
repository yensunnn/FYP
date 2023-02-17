import numpy as np
import pandas as pd
import tensorflow as tf
#from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot as plt
import datetime

date = datetime.datetime.now().strftime("%d%m%Y_%H%M")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

'''
similar to model4
MSE is calculated via dataset - not batch size 8
(ie MSE1 = criterion(all dataset1 y_hat, all dataset actual y-values))
include shuffling
'''


#creating combined dataframe from all 4 datasets
train_x1 = np.load ("npy inputs\lc50dm_train_x_ssl.npy")
train_x1 = np.float32(train_x1)
df1 = pd.DataFrame(data=train_x1)
df1['dataset'] = 1

train_x2 = np.load ("npy inputs\LD50_train_x_ssl.npy")
train_x2 = np.float32(train_x2)
df2 = pd.DataFrame(data=train_x2)
df2['dataset'] = 2

train_x3 = np.load ("npy inputs\IGC50_train_x_ssl.npy")
train_x3 = np.float32(train_x3)
df3 = pd.DataFrame(data=train_x3)
df3['dataset'] = 3

train_x4 = np.load ("npy inputs\lc50_train_x_ssl.npy")
train_x4 = np.float32(train_x4)
df4 = pd.DataFrame(data=train_x4)
df4['dataset'] = 4

df = pd.concat([df1,df2,df3,df4])

train_y1 = pd.read_csv("toxicity dataset (csv and mol)\lc50dm\lc50dm_train.csv")
train_y1 = train_y1['label']
train_y1 = np.float32(train_y1)
dfy1 = pd.DataFrame(data=train_y1)

train_y2 = pd.read_csv("toxicity dataset (csv and mol)\ld50\ld50_train.csv")
train_y2 = train_y2['label']
train_y2 = np.float32(train_y2)
dfy2 = pd.DataFrame(data=train_y2)

train_y3 = pd.read_csv("toxicity dataset (csv and mol)\igc50\igc50_train.csv")
train_y3 = train_y3['label']
train_y3 = np.float32(train_y3)
dfy3 = pd.DataFrame(data=train_y3)

train_y4 = pd.read_csv("toxicity dataset (csv and mol)\lc50\lc50_train.csv")
train_y4 = train_y4['label']
train_y4 = np.float32(train_y4)
dfy4 = pd.DataFrame(data=train_y4)

dfy = pd.concat([dfy1,dfy2,dfy3,dfy4])
dfy.columns = ['target']

#combining X and y and shuffling them
df_com = pd.concat([df, dfy], axis=1)
shuffled = df_com.sample(frac=1)

#train X will be our predictor variables
train_X = shuffled.iloc[:, 0:512]
train_X = train_X.values.tolist()
train_X = np.array(train_X)
train_X = torch.from_numpy(train_X)

#ds are the labels to show which datatsets they are from
ds = shuffled.iloc[:, 512]
ds = ds.values.tolist()
ds = np.array(ds)
ds = torch.from_numpy(ds)

#train y will be our response variables
train_y = shuffled.iloc[:, 513]
train_y = train_y.values.tolist()
train_y = np.array(train_y)
train_y = torch.from_numpy(train_y)


class Build_Data(Dataset):    
    # Constructor
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z
        self.len = self.x.shape[0]        
    # Getting the data
    def __getitem__(self, index):    
        return self.x[index], self.y[index], self.z[index]    
    # Getting length of the data
    def __len__(self):
        return self.len

train_dataset = Build_Data(train_X, train_y, ds)
train_data_iter = DataLoader(dataset = train_dataset, shuffle=True)

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

criterion = nn.MSELoss()


optimizer1 = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
optimizer2 = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

def extract(lst,num):
    return [item[num] for item in lst]

num_epoch = 5 #2000
count_epoch = 0
loss_list = []

#summary
#every epoch, we will do the following for each task:
#obtain all the predictions for task n and compare it against the actual values to get the MSE for task n
#multiply it with the number of molecules per task
#sum up this product for task 1 to 4

for epoch in range(num_epoch):
    count_epoch += 1
    print(count_epoch)
    count_data_iter = 0
    output1 = []
    output2 = []
    output3 = []
    output4 = []
    actual1 = []
    actual2 = []
    actual3 = []
    actual4 = []

    if epoch <= 1000:
        optimizer = optimizer1
    else:
        optimizer = optimizer2
    for X,y,z in train_data_iter:
        X = X.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        z = z.to(dtype=torch.float32)
        count_data_iter+=1
        #print("X:",X) #x values is 512 variables
        #print("y:",y) #y value is just 1 value, the toxicity
        optimizer.zero_grad()
        output_array = net(X)
        if z == 1: #if from dataset 1
            temp = extract(output_array,0)[0] #we will take the first out of the 4 outputs
            output1.append(temp) #append it to a list with all the dataset 1 predictions
            actual1.append(y) #append it to a list with the actual dataset 1 y values
        elif z == 2:
            temp = extract(output_array,1)[0]
            output2.append(temp)
            actual2.append(y)
        elif z == 3:
            temp = extract(output_array,2)[0]
            output3.append(temp)
            actual3.append(y)
        elif z == 4:
            temp = extract(output_array,3)[0]
            output4.append(temp)
            actual4.append(y)

    actual1 = torch.FloatTensor(actual1)   #convert the list into a tensor
    output1 = torch.FloatTensor(output1)
    actual2 = torch.FloatTensor(actual2)
    output2 = torch.FloatTensor(output2)
    actual3 = torch.FloatTensor(actual3)
    output3 = torch.FloatTensor(output3)
    actual4 = torch.FloatTensor(actual4)
    output4 = torch.FloatTensor(output4)
    loss1 = criterion(actual1,output1)    #get the MSE using ALL the dataset 1 predictions and actual y values
    loss1 *= len(df1) #multiply with the dataset1 size
    loss2 = criterion(actual2,output2)
    loss2 *= len(df2)
    loss3 = criterion(actual3,output3)
    loss3 *= len(df3)
    loss4 = criterion(actual4,output4)
    loss4 *= len(df4)
    loss = loss1+loss2+loss3+loss4    #sum up for all the 4 tasks
    sumloss = Variable(loss.data, requires_grad=True)
    loss_list.append(loss)
    sumloss.backward()
    optimizer.step()

step = np.linspace(0, num_epoch, len(loss_list))
fig, ax = plt.subplots(figsize=(8,5))
epochs = range(1,num_epoch)
plt.plot(step, np.array(loss_list))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

'''
PATH = 'MTDNN_ssl_statedict_MSE_' + date + '.pt'
torch.save(net.state_dict(), PATH)
PATH1 = 'MTDNN_ssl_MSE_' + date + '.pt'
torch.save(net, PATH1)'''


