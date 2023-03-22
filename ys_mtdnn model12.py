import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime

date = datetime.datetime.now().strftime("%d%m%Y_%H%M")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

'''
key details of this model:
    after talking to JJ
    1. one input gives one output
    2. changing up the loss function to calculate loss per ds then sum up and minimize the sum loss
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

#min max normalizing
normalized_df=(df-df.min())/(df.max()-df.min())
normalized_df['dataset'] = df['dataset']

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

#min max normalizing
normalized_df_y=(dfy-dfy.min())/(dfy.max()-dfy.min())

#combining X and y and shuffling them
df_com = pd.concat([normalized_df, normalized_df_y], axis=1)
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
train_data_iter = DataLoader(dataset = train_dataset, batch_size = 8, shuffle=True)

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()    #512 (1024 512 512 512) 1
       self.input_layer = nn.Sequential(torch.nn.Linear(512, 1024),nn.ReLU(), nn.BatchNorm1d(1024))
       self.layer1 = nn.Sequential(torch.nn.Linear(1024, 512),nn.ReLU(),nn.BatchNorm1d(512))
       self.layer2 = nn.Sequential(torch.nn.Linear(512, 512),nn.ReLU(),nn.BatchNorm1d(512))
       self.layer3 = nn.Sequential(torch.nn.Linear(512, 512),nn.ReLU(),nn.BatchNorm1d(512))
       self.output_layer = torch.nn.Linear(512,1)
   
   def forward(self, x):
       x = self.input_layer(x)
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.output_layer(x)      
       return x

net = Net()
criterion = nn.MSELoss()

optimizer1 = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5) #0.01
optimizer2 = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

def extract(lst,num):
    return [item[num] for item in lst]

def resultLoss(actual_arr, output_arr, length):
    actual_arr = torch.FloatTensor(actual_arr)
    actual_arr = actual_arr.view(-1,1)
    output_arr = torch.FloatTensor(output_arr)
    output_arr = output_arr.view(-1,1)
    output_arr = Variable(output_arr.data, requires_grad=True)
    loss = criterion(output_arr,actual_arr)
    loss *= length
    return loss
    
num_epoch = 500 #2000
trainingEpoch_loss = []

#epochs:
#every epoch, we will do the following for each task:
#obtain all the predictions for task n and compare it against the actual values to get the MSE for task n
#sum up this product for task 1 to 4
for epoch in range(num_epoch):
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
        for idx,ds in enumerate(z):
            temp = extract(output_array,0)[idx]
            if ds == 1: #if from dataset 1
                output1.append(temp) #append output to a list with all the dataset 1 predictions
                actual1.append(y[idx]) #append target to a list with the actual dataset 1 y values
            elif ds == 2:
                output2.append(temp)
                actual2.append(y[idx])
            elif ds == 3:
                output3.append(temp)
                actual3.append(y[idx])
            elif ds == 4:
                output4.append(temp)
                actual4.append(y[idx])
    
    loss1 = resultLoss(actual1, output1, len(df1))
    loss2 = resultLoss(actual2, output2, len(df2))
    loss3 = resultLoss(actual3, output3, len(df3))
    loss4 = resultLoss(actual4, output4, len(df4))
    training_loss = (loss1+loss2+loss3+loss4)    #sum up losses from all the 4 tasks
    print(epoch, training_loss)
    training_loss.backward()
    optimizer.step()
    trainingEpoch_loss.append(training_loss.item())


#plotting the training loss
plt.plot(trainingEpoch_loss, label='train_loss')
plt.legend()
plt.show()


