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
key details of this model 10:
shuffled the inputs both upon importing and also in DataLoader
min-max normalization applied [0,1]
batch normalization applied
loss function is same as eqn 9 (explained in teams)

parts of the model that were mentioned by the original author:
learning rate and momentum of the SGD optimizer
number of node & layers in NN 
number of epochs used
batch size = 8

things i have tried but did not help:
initializing weights
adding learning rate scheduler (linear and exponential)
scaling data to [-1,1]
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
       self.output_layer = torch.nn.Linear(512,4)
   
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
    
    
num_epoch = 2000 #2000
count_epoch = 0
loss_list_tensor = []

#epochs:
#every epoch, we will do the following for each task:
#obtain all the predictions for task n and compare it against the actual values to get the MSE for task n
#sum up this product for task 1 to 4
for epoch in range(num_epoch):
    count_epoch += 1
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
            if ds == 1: #if from dataset 1
                temp = extract(output_array,0)[idx] #we will take the first out of the 4 outputs
                output1.append(temp) #append output to a list with all the dataset 1 predictions
                actual1.append(y[idx]) #append target to a list with the actual dataset 1 y values
            elif ds == 2:
                temp = extract(output_array,1)[idx]
                output2.append(temp)
                actual2.append(y[idx])
            elif ds == 3:
                temp = extract(output_array,2)[idx]
                output3.append(temp)
                actual3.append(y[idx])
            elif ds == 4:
                temp = extract(output_array,3)[idx]
                output4.append(temp)
                actual4.append(y[idx])
            #print(idx, output_array, ds, temp)
    
    loss1 = resultLoss(actual1, output1, len(df1))
    loss2 = resultLoss(actual2, output2, len(df2))
    loss3 = resultLoss(actual3, output3, len(df3))
    loss4 = resultLoss(actual4, output4, len(df4))
    sumloss = (loss1+loss2+loss3+loss4)    #sum up lossed from all the 4 tasks
    print(count_epoch, sumloss)
    loss_list_tensor.append(sumloss)
    sumloss.backward()
    optimizer.step()

#plotting the training loss
loss_list = []
for x in loss_list_tensor:
    x =  x.detach().numpy()
    loss_list.append(x)
    
step = np.linspace(0, num_epoch, len(loss_list))
fig, ax = plt.subplots(figsize=(8,5))
epochs = range(1,num_epoch)
plt.plot(step, loss_list)
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

#saving model
PATH = 'MTDNN_ssl_statedict_MSE_' + date + '.pt'
#torch.save(net.state_dict(), PATH)
PATH1 = 'MTDNN_ssl_MSE_' + date + '.pt'
#torch.save(net, PATH1)

print('done')

def MA(arr, window_size):
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
      
    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:
        
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]
      
        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
          
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
          
        # Shift window to right by one position
        i += 1
    return moving_averages