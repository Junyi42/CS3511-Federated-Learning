import socket
import dill
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_local_model(train_loader,global_model_state_dict,epochs,learning_rate,num_classes):
    local_model = MLP()
    local_model.load_state_dict(global_model_state_dict)
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            one_hot_labels = F.one_hot(labels, num_classes).float()
            #one_hot_labels = torch.FloatTensor(one_hot_labels)
            probs = local_model(features)
            loss = criterion(probs, one_hot_labels)
            loss.backward()
            # update model parameters
            optimizer.step()
            
    return local_model.state_dict()

def load_client_dataset(batch_size):
    #load data
    train_dataset_clients = []
    for i in range(20):
        with open("./Client"+str(i+1)+".pkl",'rb') as f:
            train_dataset_clients.append( dill.load(f))
            
    dataloader_client = []
    for dataset in train_dataset_clients:
        dataloader_client.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    return dataloader_client

def load_test_dataset(batch_size):
    # load test data
    test_dataset = torchvision.datasets.MNIST(
            './data', train=False, download=True, 
            transform=transforms.ToTensor())

    dataloader_test = DataLoader(test_dataset, batch_size, shuffle=False)
    return dataloader_test

def agggregate_local_models(local_model_state_dicts):
    avg_model = MLP()
    for i in range(len(local_model_state_dicts)):
        client_model = MLP()
        client_model.load_state_dict(local_model_state_dicts[i])
        for param_avg, param_client in zip(avg_model.parameters(), client_model.parameters()):
            if i==0:
                param_avg.data = param_client.data
            else:
                param_avg.data = param_avg.data + param_client.data

    for param_avg in avg_model.parameters():
        param_avg.data /= len(local_model_state_dicts)
        
    return avg_model.state_dict()

def test_global_model(model, test_loader):
    model.eval()
    final_predictions = []
    final_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            preds = model(features)
            predicted_classes = torch.argmax(preds, dim=-1)
            predicted_classes = predicted_classes.tolist()
            labels = labels.tolist()
            final_predictions.extend(predicted_classes)
            final_labels.extend(labels)

    final_predictions = torch.tensor(final_predictions)
    final_labels = torch.tensor(final_labels)

    accuracy = (final_predictions == final_labels).float().mean().item()
    print('Global Model Accuracy: {:.2f}%'.format(accuracy * 100))
    return accuracy
