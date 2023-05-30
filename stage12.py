# %%
import socket
import numpy as np
from utils import MLP
import dill
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import multiprocessing
from torch.utils.data import Dataset, DataLoader

# %%
#load data
train_dataset_clients = []
for i in range(20):
    with open("./Client"+str(i+1)+".pkl",'rb') as f:
        train_dataset_clients.append( dill.load(f))


# %%
# load test data
test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, 
        transform=transforms.ToTensor()
    )

test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# %%
print("len(train_dataset_clients):",len(train_dataset_clients)) # 20 clients
print("len(train_dataset_clients[0]):",len(train_dataset_clients[0])) # 3000 (image,label)s/client
print("len(train_dataset_clients[0][0]):",len(train_dataset_clients[0][0])) #  (image, label)
print("train_dataset_clients[0][0][0].shape",train_dataset_clients[0][0][0].shape) # image shape [1, 28, 28]

# %%
batch_size = 32
dataloader_client = []
for dataset in train_dataset_clients:
    dataloader_client.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
print(len(dataloader_client[0]))


# %% [markdown]
# ### Stage 1/2: all/M clients participate in each round of updates
# we suppose that one server and N/M clients train a classification task collaboratively, which updates model parameters by 
# a) direct data access or b) “.pth” file reading and writing. Here we choose **b) “.pth” file reading and writing**

# %%
num_classes = 10
num_epochs = 5
lr_client = 0.01
num_rounds = 10
import time
from tqdm import tqdm
def train(num_rounds,num_epochs,lr,mode='all',M=20):
    # 初始化全局模型参数
    start_time = time.time()
    global_model = MLP()
    global_model_path = './models/server_model.pth'
    torch.save(global_model.state_dict(), global_model_path)
    for r in (range(num_rounds)):
        # train M clients
        if mode == 'all':
            idx = [i for i in range(len(dataloader_client))]
        else: # mode =='partial'
            idx = np.random.choice(len(dataloader_client), M, replace=False)
        for i in tqdm(idx): # for each client
            model = MLP() 
            # load .pth file
            checkpoint = torch.load(global_model_path)
            model.load_state_dict(checkpoint)
            # load the corresponding dataloader
            dataloader = dataloader_client[i]
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr)
            for e in range(num_epochs):
                for features, labels in dataloader:
                    optimizer.zero_grad()
                    one_hot_labels = F.one_hot(labels, num_classes).float()
                    #one_hot_labels = torch.FloatTensor(one_hot_labels)
                    probs = model(features)
                    loss = criterion(probs, one_hot_labels)
                    loss.backward()
                    # update model parameters
                    optimizer.step()

            # 保存模型参数到.pth文件
            save_path = "./models/client_models/model"+str(i+1)+'.pth' # e.g. ./models/client_models/model1.pth ~ model20.pth
            torch.save(model.state_dict(), save_path)


        print("Round",r+1,"finished")
        # 计算全局模型参数平均值
        avg_model = MLP()
        avg_model.load_state_dict(torch.load(global_model_path))
        for i in idx:
            client_model_path ="./models/client_models/model"+str(i+1)+'.pth' # e.g. ./models/client_models/model1.pth ~ model20.pth
            client_model = MLP()
            client_model.load_state_dict(torch.load(client_model_path))
            for param_avg, param_client in zip(avg_model.parameters(), client_model.parameters()):
                param_avg.data = param_avg.data + param_client.data

        for param_avg in avg_model.parameters():
            param_avg.data /= len(dataloader_client)

        # 更新全局模型参数
        global_model.load_state_dict(avg_model.state_dict())
        torch.save(global_model.state_dict(), global_model_path)


# %%
def test(global_model): 
    
    global_model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:

            preds = global_model(features)
            predicted_classes = torch.argmax(preds, dim=1)
            
            predicted_classes = predicted_classes.tolist()
            labels = labels.tolist()

            all_predictions.append(predicted_classes)
            all_labels.append(labels)

    # 将所有预测结果和实际标签转换为Tensor类型
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)

    accuracy = (all_predictions == all_labels).float().mean().item()
    print('Global Model Accuracy: {:.2f}%'.format(accuracy * 100))

# %%
print("train on stage 1")
r = 10
e = 5
lr = 0.01
train(r,e,lr)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global_model = MLP()
global_model_path = './models/server_model.pth'
global_model.load_state_dict(torch.load(global_model_path))
test(global_model)


# %%
#Stage 2:

print("train on stage 2")
r = 10
e = 5
lr = 0.01
m = 5
for e in [10,15,20,25]:
    print("r:",r,"e:",e,"m:",m,"lr:",lr)
    train(r,e,lr,mode='partial',M=m)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model = MLP()
    global_model_path = './models/server_model.pth'
    global_model.load_state_dict(torch.load(global_model_path))
    test(global_model)
